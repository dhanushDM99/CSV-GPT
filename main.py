# using streamlit for UI
import streamlit as st  

from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
import langchain_openai

from dotenv import load_dotenv
import os


def main():
    load_dotenv()

    # Loading the API key from the environment variable
    # and instructing the user whether the API key is set or not..
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("Please set the OPENAI_API_KEY !")
        exit(1)
    else:
        print("OPENAI_API_KEY is set!")

       
    st.set_page_config(page_title="KnowWize CSV-GPT")
    st.title("CSV-GPT")
    st.divider()

    # named the uploaded csv docs by user as csv_doc
    csv_doc = st.file_uploader("Upload!", type="csv") 

    # temperature = 0 for constant and accurate answers 
    # using verbose = True ,for printing its thinking process
    if csv_doc is not None:
        agent = create_csv_agent(
            langchain_openai.OpenAI(temperature=0), csv_doc, verbose=True)
        
        # this prompt is written for only this particular use case(dataset or project)
        # other data sets and projects may require different prompts 
        # and its also possible to write generalised prompt fo projects similar to this..

        prompt = """The following lines describe the details of the given csv document,with the columns  described as 
        
        column STATUS : represents  whether the order is Shipped , Cancelled , Resolved , On Hold ,In Process.
        column QTR_ID : represents the ID of the quarter in which order took place like quarter 1 or 2 or 3 or 4 in a year . 
        column MSRP : MSRP stands for Manufacturer's Suggested Retail Price .

        and the remaining columns has their usual meaning .When a question is asked please assume yourself 
        as a data scientist which means you need to use relevant mathematical tools , skills , techniques
        to answer asked queries and also should be able to correctly perform questions like aggregation, pivot 
        table like queries and etc….
        """

        user_question = st.text_input("Ask about given CSV doc: ")
        st.caption('Please explain the question clearly for complex queries..')
        
        #appending the written prompt the user's question
        final_question = prompt + user_question

        if user_question is not None and user_question != "":
                with st.spinner(text="Thinking..."):
                    st.write(agent.run(final_question))







if __name__ == "__main__":
    main()
