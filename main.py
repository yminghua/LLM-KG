from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import GraphCypherQAChain
from create_KG import extract_and_store_graph
from tqdm import tqdm
import yaml
import os
import random
import argparse
import sys
import time
from datetime import datetime
import openai
import tiktoken
from LEval_utils import *


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
neo4j_config = config['neo4j_config']
openai_api_key = config['openai_api_key']


def parse_arguments():
    parser = argparse.ArgumentParser(description='LLM-KG Construction and Querying Tool')
    
    parser.add_argument('--metric', choices=["llm_turbo_eval","llm_gpt4_eval","exam_eval", "ngram_eval", "human_eval"], required=True, help='metric name from ["turbo_eval","gpt4_eval","auto_eval", ...]')

    parser.add_argument('--max_length', default="16k", help='max length of the input, e.g., 2k, 16k')
    # if none, we will load from huggingface
    parser.add_argument('--task_path', type=str, default=None, help= 'set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None, help='optional, if not set, we will test all. set this if you want test a specific task from huggingface, example: coursera, tpo')
    parser.add_argument('--mc_tasks', action='store_true')
    
    # Demo flag argument
    parser.add_argument('--demo', action='store_true', help='Run the program in demo mode with predefined settings')
    
    return parser.parse_args()


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_time():
    # Get the current date and time
    now = datetime.now()

    # Format the output as 'Month-Day-Year Hour:Minute:Second'
    formatted_date_time = now.strftime("%m-%d-%Y %H:%M:%S")

    # Parse the original string to a datetime object
    datetime_obj = datetime.strptime(formatted_date_time, "%m-%d-%Y %H:%M:%S")

    # Format the datetime object to the desired format
    formatted_str = datetime_obj.strftime("%m-%d-%H-%M")

    return formatted_str


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
        
        
def generate_KG(graph, llm, document):
    
    raw_documents = Document(document, {})
    
    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    documents = text_splitter.split_documents([raw_documents])

    # Clean up graph database
    graph.query("MATCH (n) DETACH DELETE n")

    # Specify which node labels should be extracted by the LLM
    allowed_nodes = ["Person", "Company", "Location", "Event", "Movie", "Service", "Award"]

    print("\nConstructing KG...")
    for i, d in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(graph, llm, d, allowed_nodes)
        
        
def query_KG(graph, query):
    
    graph.refresh_schema()

    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=ChatOpenAI(temperature=0, model="gpt-4"),
        qa_llm=ChatOpenAI(temperature=0.0001, model="gpt-3.5-turbo"),
        validate_cypher=True, # Validate relationship directions
        verbose=False
    )
    
    try:
        instruction_prompt = "Choose one option to answer the following question (only provide your chosen option): "
        results = cypher_chain.invoke(instruction_prompt + query)
        return results['result']
        
    except Exception as e:
        # exception_type = type(e).__name__
        error_message = str(e)
        e_type = "Generated Cypher Statement is not valid"
        if e_type in error_message:
            return "NOANSWER"
        else:
            return "RETRY"


def main():
    
    args = parse_arguments()
    
    # Connect to Neo4j Database
    graph = Neo4jGraph(
        url=neo4j_config['url'],
        username=neo4j_config['username'],
        password=neo4j_config['password']
    )
    
    # Get access to LLM (chatgpt)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    
    if args.demo:
        print("Running in demo mode...")
        
        # Extract the wikipedia article
        print("Extracting articles about 'Taylor Swift' from Wikipedia...")
        raw_documents = WikipediaLoader(query="Taylor Swift").load()
        
        # Define chunking strategy
        text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)

        # Only take the first three documents
        print(f"Find {len(raw_documents)} articles, only use the first three for demo...")
        documents = text_splitter.split_documents(raw_documents[:3])
        
        # Clear up the graph
        graph.query("MATCH (n) DETACH DELETE n")
        
        # Specify which node labels should be extracted by the LLM
        allowed_nodes = ["Person", "Company", "Location", "Event", "Movie", "Service", "Award"]

        print("Constructing KG...")
        for i, d in tqdm(enumerate(documents), total=len(documents)):
            extract_and_store_graph(graph, llm, d, allowed_nodes)
                    
        # Perform Q&A Test
        print("\nPerform Q&A Test...")
        graph.refresh_schema()

        cypher_chain = GraphCypherQAChain.from_llm(
            graph=graph,
            cypher_llm=ChatOpenAI(temperature=0, model="gpt-4"),
            qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            validate_cypher=True, # Validate relationship directions
            verbose=False
        )

        query = "When was Taylor Alison Swift born?"
        results = cypher_chain.invoke(query)
        print(f"Query: {query}")
        print(f"Result: {results['result']}")
        
        query = "Which album does the song 'Shake It Off' belong to?"
        results = cypher_chain.invoke(query)
        print(f"\nQuery: {query}")
        print(f"Result: {results['result']}")
        
        query = "Taylor Swift's hometown is in Illinois, True or False?"
        results = cypher_chain.invoke(query)
        print(f"\nQuery: {query}")
        print(f"Result: {results['result']}")
            
    else:
        key_data_pairs = {}

        max_length = k_to_number(args.max_length) - max_new_tokens
        openai_model = "turbo-" + args.max_length + '-' + get_time()
        data_save_path = f"Predictions/{args.metric}/{openai_model}"
        input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")
        build_key_data_pairs(args, key_data_pairs, data_save_path)
        
        start_idx = 0
        for file_name in key_data_pairs:
            sys_prompt = get_sys_prompt(args, file_name)
            fw = open(f'{file_name}', "w")
            data = key_data_pairs[file_name]
            for d in tqdm(data):
                document = d['input']
                generate_KG(graph, llm, document)
                cnt = 0
                while num_tokens_from_string(document, "gpt-3.5-turbo") > max_length:
                    if "code" not in file_name:
                        document = " ".join(document.split(" ")[:max_length - cnt]) # chunk the input len from right
                    else:
                        document = " ".join(document.split(" ")[cnt - max_length:]) # chunk the input len from left
                    cnt += 250
                
                print('document len', num_tokens_from_string(document, "gpt-3.5-turbo"))

                instructions = d['instructions']
                outputs = d['outputs']
                i = 0

                for inst, out in zip(instructions, outputs):
                    messages = [{"role": "system", "content" : sys_prompt}]
                    save_d = {}
                    save_d['query'] = inst
                    save_d['gt'] = out
                    if "gsm" in file_name or "codeU" in file_name:
                        messages.append({"role": "user", "content": document + "\n\n" + inst})
                        save_d['prompt'] = sys_prompt + inst

                    elif args.metric == "exam_eval":
                        context = "Document is as follows. {} Question: {} \nPlease directly give answer without any additional output or explanation\n Answer: "
                        messages.append({"role": "user", "content": context.format(document, inst)})
                        save_d['prompt'] = sys_prompt + context
                    else:
                        context = "Document is as follows. {} Instruction: {} " + f"The suggested output length is around {len(out.split())} words. Output: "
                        messages.append({"role": "user", "content": context.format(document, inst)})
                        save_d['prompt'] = sys_prompt + context

                    for _ in range(3):
                        try:
                            if start_idx == 0:
                                print(messages[1]["content"])
                                print("--------------------------- end of example input ------------------")
                                input("Press Enter to confirm this is the correct input for the api call ...")
                                start_idx += 1
                                
                            if d['source'] == 'toefl_tpo':
                                ret = query_KG(graph, inst)
                                if ret == 'NOANSWER' or ret[:2] not in ['A.', 'B.', 'C.', 'D.']:
                                    options = ['A', 'B', 'C', 'D']
                                    ret = random.choice(options)
                                if ret == 'RETRY':
                                    continue
                                ret = ret.strip()
                                save_d[f'{openai_model}_pred'] = ret[0]
                                save_d['evaluation'] = d['evaluation']
                                
                            else:
                                response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo-16k-0613",
                                    messages=messages, 
                                    max_tokens=max_new_tokens,
                                    temperature=0.0001,
                                )  # get response
                                ret = response['choices'][0]['message']['content']
                                ret = ret.strip()  # get the paraphrased answer

                                save_d[f'{openai_model}_pred'] = ret
                                save_d['evaluation'] = d['evaluation']

                            # test the factuality in scientific fiction
                            if "sci_fi" in file_name:
                                text_inputs = inst.replace("based on the world described in the document.",
                                                        "based on the real-world knowledge and facts up until your last training") + "\nPlease directly give answer without any additional output or explanation \nAnswer:"
                                messages.append({"role": "user", "content": text_inputs})
                                response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo-16k-0613",
                                    messages=messages,
                                    max_tokens=max_new_tokens,
                                    temperature=0.0001,
                                )  # get response
                                ret = response['choices'][0]['message']['content']
                                ret = ret.strip()  # get the paraphrased answer
                                save_d[f'{openai_model}_pred'] += f" [fact: {ret}]"

                            print("----------------- [output] vs [ground truth] -----------------")
                            print('[output]:', save_d[f'{openai_model}_pred'], "\n" , '[ground truth]:', save_d['gt'], "\n")
                            fw.write(json.dumps(save_d) + '\n')
                            break

                        except Exception as e:  # add some logic here for retry
                            if isinstance(e, KeyboardInterrupt):
                                raise e
                            print(i, e)

                            time.sleep(0.8)
                    time.sleep(1.0)
                    i += 1
                    
            fw.close()



if __name__ == "__main__":
    sys.exit(main())
