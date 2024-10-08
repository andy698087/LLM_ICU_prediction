# Importing necessary libraries
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into smaller chunks
from langchain_chroma import Chroma  # For using Chroma as a vector store

from langchain_openai.chat_models import ChatOpenAI  # To interact with OpenAI's chat models
from langchain_openai.embeddings import OpenAIEmbeddings  # For creating text embeddings

from langchain_core.prompts import ChatPromptTemplate  # For creating structured prompts
from pydantic import BaseModel, Field  
from langchain_core.output_parsers import StrOutputParser  # To parse outputs from the model
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph

import os 
import numpy as np

# Dictionaries for translating anesthesia types and department names to descriptive English text
dict_anestype = {
    'ETGA': 'general anesthesia',
    'SA': 'spinal anesthesia',
    'IVGA': 'monitored anesthesia care',
    'Mask GA(LMA)': 'general anesthesia using supraglottic airway',
    'EA': 'epidural anesthesia'
}

dict_department = {
    '婦產部': 'Obstetrics and Gynecology Department',
    '骨科部': 'Orthopedic Department',
    '一般外科': 'General Surgery',
    '心臟血管外科': 'Cardiovascular Surgery',
    '創傷科': 'Trauma Surgery',
    '心臟血管內科': 'Cardiovascular Medicine',
    '耳鼻喉科暨頭頸外科': 'Otolaryngology and Head & Neck Surgery',
    '神經外科': 'Neurosurgery',
    '整形外科': 'Plastic Surgery',
    '小兒外科': 'Pediatric Surgery',
    '泌尿科': 'Urology Department',
    '形體美容醫學中心': 'Cosmetic and Body Contouring Center',
    '口腔顎面外科': 'Oral and Maxillofacial Surgery',
    '麻醉部': 'Anesthesiology Department',
    '人工耳蝸中心': 'Cochlear Implant Center',
    '眼科部': 'Ophthalmology Department',
    '小兒部': 'Pediatrics Department',
    '大腸直腸外科': 'Colorectal Surgery',
    '胸腔外科': 'Thoracic Surgery',
    '影像醫學科': 'Radiology Department',
    '肝膽胃腸科': 'Hepatobiliary and Gastroenterology Department',
    '牙科部': 'Dental Department'
}

# Load the test samples dataset
# Columns included: ['術前診斷', '排程手術名稱', '手術部門', 'op_type (urgency level of surgery)', 'anes_type (anesthesia type)', '這次入院診斷', '這次主訴', '這次現在病史','先前診斷_1', '先前處置_1', '先前診斷_2', '先前處置_2', '先前診斷_3', '先前處置_3','ICU_24h', 'ICU_48']
test_samples = pd.read_pickle('test_samples1000_20240929.pkl')
print(test_samples.shape)  # Displaying the shape of the dataset
test_samples['ICU_24'].value_counts()  # Displaying the counts of values in the ICU_24 column
print(test_samples.head())  # Displaying the first few rows of the dataset
print(test_samples.columns)  # Displaying the column names of the dataset

# Set up the OpenAI API key for use
OPENAI_API_KEY = "YOUR_API_KEY"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Store the API key in the environment

# Define the embedding model that will convert text chunks into vectors
embeddings = OpenAIEmbeddings()

# Define a directory for storing the Chroma vectorstore
persist_directory = "chroma_vectorstore_OpenAI"  # You can change this directory name as needed

# Create the directory if it does not already exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

  # 使用Chroma建立vectorstore，並將其轉為retriever型態
  vectorstore = Chroma.from_documents(
    documents=doc_split,
    embedding=embeddings,
    persist_directory=persist_directory
  )

  # Save the vectorstore to disk
  vectorstore.persist()


  # After processing all documents, persist to disk
  vectorstore.persist(persist_directory=persist_directory)

else:
  # If existed, load and initialize the Chroma vectorstore with the specified embedding model
  vectorstore = Chroma(
      persist_directory=persist_directory,  # Specify the directory to store the vectorstore
      embedding_function=embeddings  # Use the OpenAI embeddings for vector generation
  )

# Set up the vectorstore as a retriever for document retrieval tasks
retriever = vectorstore.as_retriever()
print("finished loading retriever")  # Confirm the retriever is ready

# Define the directory for saving the output results
output_directory = 'LLM_results_resample1000_20240929'

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Prompt template for instructing the language model
instruction = """
You are an assistant responsible for handling user queries.
You are given a task and context.
The context contains information from the surgical procedure and patient's medical record.
Please use the extracted document content to assess the patient in the context of the proposed procedure and then provide an answer.
If the answer cannot be obtained from the documents, reply directly that you do not know, and do not fabricate answers.
Note: Ensure the accuracy of the response.

[Task]
Your task is to generate answers to the user’s question.
Please think step by step and give your response in JSON format using the provided template.
The desired response type is provided in angle brackets < >.

<CLS>: Provide a classification response from the following 5 levels:
	•	"1" (Strongly Disagree)
	•	"2" (Disagree)
	•	"3" (Neutral)
	•	"4" (Agree)
	•	"5" (Strongly Agree)

<str>: Provide a textual explanation or description.

JSON Template:
{
"Step By Step Explanation": "<str>",
"Answer": "<CLS>"
}
"""

# Create the prompt template using the instruction defined above
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("system", "document: \n\n {documents}"),
        ("human", "question: {question}"),
    ]
)

# Function to generate ICU predictions using a retrieval-augmented generation (RAG) chain
import json

def generate_ICU_prediction(row, rag_chain):
    question = f"""
    Task: Should postoperative intensive care be required for this patient?

    Context:
    Provider Service: {dict_department[row['手術部門']]}
    Procedure Description: {row['排程手術名稱']}
    Diagnosis: {row['術前診斷']}
    Urgency level: {row['op_type']}
    Planned anesthesia: {row['anes_type']}

    Current Medical Record Notes:
    Chief complaint: {row['這次主訴']}
    Present illness: {row['這次現在病史']}

    Previous Medical History:
    Discharge Diagnosis 1: {row['診斷_1']}
    Discharge treatment 1: {row['處置_1']}
    Discharge Diagnosis 2: {row['診斷_2']}
    Discharge treatment 2: {row['處置_2']}
    Discharge Diagnosis 3: {row['診斷_3']}
    Discharge treatment 3: {row['處置_3']}
    """
    docs = retriever.invoke(question)  # Retrieve relevant documents using the retriever
    generation = rag_chain.invoke({"documents": docs, "question": question}).replace("```", "").replace("json\n", "")
    try:
        generation = json.loads(generation)  # Parse the response as JSON
    except:
        print(generation)
        generation = {'Step By Step Explanation': generation, 'Answer': 'unknown'}
    return generation['Answer'], generation['Step By Step Explanation']

# Function to apply the RAG chain to the dataset and save the output results
def RAG_chain_output(data, list_models=['gpt-4-turbo-2024-04-09']):
    for model in list_models:
        print(f"start running {model}")
        for n_chunk in range(round(len(data) / 200) + 1):
            output_file = f'ICU_prediction_20240928_noStructText_{model}_{n_chunk + 1}.csv'
            if output_file not in os.listdir(output_directory):
                slice_start = 200 * n_chunk
                if n_chunk == round(len(data) / 200) + 1:
                    slice_end = None
                else:
                    slice_end = 200 * (n_chunk + 1)

                # Set up the language model and chain
                llm = ChatOpenAI(model=model, temperature=0)
                rag_chain = prompt | llm | StrOutputParser()
                data_sliced = data[slice_start:slice_end]
                data_sliced[['ICU_prediction', 'ICU_prediction_reason']] = data_sliced.apply(generate_ICU_prediction, args=(rag_chain,), axis=1, result_type="expand")
                col_output = ['ICU_24', 'ICU_prediction', 'ICU_prediction_reason']
                print(f'{data_sliced[col_output].head()}')

                # Save the output results to a CSV file
                output_dir = os.path.join(output_directory, output_file)
                data_sliced[col_output].to_csv(output_dir)
                print(f'Results have been saved to {output_dir}')

# Run the RAG chain output function with the test samples
RAG_chain_output(test_samples)

quit()
