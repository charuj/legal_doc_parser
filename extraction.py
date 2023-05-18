# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain


# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader

# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime
import os 
import openai

# Text Helpers
from bs4 import BeautifulSoup
from typing import List, Optional


# For token counting
from langchain.callbacks import get_openai_callback

    
class DocumentLoader:
    def __init__(self, file_path):
        self.file_path= file_path 
    
    def load_pdf(self):
        pdfloader = PyPDFLoader(self.file_path)
        pages = pdfloader.load_and_split()
        return pages 


class formatOutput:
    def __init__(self, gpt_output):
        self.output= gpt_output 
    
    def print_output(self):
        print(json.dumps(output,sort_keys=True, indent=3))
    
    def output_table(self):
        table= pd.DataFrame(output['prenupschema'])
        return table 
    
    def output_csv(self):
        table= pd.DataFrame(output['prenupschema'])
        table.to_csv('parsed_prenup_data.csv', index=False)



class PrenupSchema(BaseModel):
    spouse_1: str = Field(
        description="The name of the first spouse mentioned in the prenuptial agreement",
    )
    spouse_2: str = Field(
        description="The name of the second spouse mentioned in the prenuptial agreement",
    )
    agreement_date: Optional[str] = Field(
        description="Date of when the agreement was signed",
    )

    marriage_date: Optional[str] = Field(
        description="Date of when the couple intends to get legally married",
    )
    state: Optional[str] = Field(
        description="State in the United States where this agreement is fomalized or state where the couple live",
    )

    spouse_1_property: Optional[str] = Field(
        description="All property mentioned in the prenuptial agreement that belongs to the first spouse. Includes money, stocks, cash, real estate, vehicles, art, jewelry etc.",
    )

    spouse_2_property: Optional[str] = Field(
        description="All property mentioned in the prenuptial agreement that belongs to the second spouse. Includes money, stocks, cash, real estate, vehicles, art, jewelry etc.",
    )



if __name__ == '__main__':

    openai.api_key = os.environ["OPENAI_API_KEY"]

    loader= DocumentLoader("sample_prenups/Premarital-Agreement_sample.pdf")
    pages = loader.load_pdf()


    schema, extraction_validator = from_pydantic(
        PrenupSchema,
        description="Extract key information from a legal prenuptial agreement between two individuals in a relationship prior to marriage",
        examples=[
            (
                "THIS AGREEMENT made this 15 day of November, 2021, by and between JANE SMITH, residing at Boulder Colorado, hereinafter referred to as “Jane” or “the Wife”, and JOHN DOE, residing at Boulder Colorado, hereinafter referred to as “John” or “theHusband. Property of JANE SMITH includes  Bank of America checking account withan approximate balance of $139,500.00, and a 2020 Mazda 3.Property of JOHN DOE includes The real property known as the Texus Ranch, an authentic Monet painting, and 100% of the funds, stocks, bonds and other assets on deposit in any investment, brokerage, money market, stock and retirement accounts standing in the name of JOHN DOE as of 15/11/2021 ;",
                {"spouse_1": "Jane Smith", "spouse_1": "John Doe", "agreement_date": "15/11/2021","spouse_1_property" : "Bank of America checking account with balance of $139,500.00, 2020 Mazda 3", "spouse_2_property" : "Texus Ranch, Monet painting,funds, stocks, bonds " },
            )
        ],
        many=True,
    )

    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        max_tokens=2000,
        openai_api_key=openai.api_key
    )

    chain = create_extraction_chain(llm, schema, input_formatter="triple_quotes")


    ##gpt-4 has a context window limit of 8192 tokens. However, the full document has 14120 tokens
    ## gpt-4-32 is in beta and has a context window of 32K, so we should eventually be able to pass the entire doc
    ## For now, we'd either need to get creative about splitting the documents or using Anthropic's API which has a larger context window 
    relevant_pages=str(pages[0])+str(pages[20])+str(pages[21])+str(pages[22])+ str(pages[23])+str(pages[24])
    output = chain.predict_and_parse(text=relevant_pages)["data"]

    return_output= formatOutput(output)
    return_output.output_csv()