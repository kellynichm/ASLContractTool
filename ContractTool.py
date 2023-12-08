import vertexai
from vertexai.language_models import TextGenerationModel
from pathlib import Path
import PyPDF2
from google.cloud import storage
import fitz
import json
import io
import base64
import streamlit as st
from PIL import Image
import pdfminer
import pdfminer.high_level
#from src.utils import * - removed as everythign is now in the single file - This line can be removed. We should also be able to remove the src/utils file.
#from src.vertex import * - removed as everything is now in the single file - This line can be removed. We should also be able to remove the src/vertex file. 

#imports from the utils.py - see above
import streamlit as st

#imports from the vertex src - see above
from vertexai.preview.language_models import TextGenerationModel
import vertexai
import streamlit as st
import os

#running utils.py directly here - See above commment

def reset_session() -> None:
    st.session_state['temperature'] = 0.0
    st.session_state['token_limit'] = 1024
    st.session_state['top_k'] = 1
    st.session_state['top_p'] = 0.1
    st.session_state['debug_mode'] = False
    st.session_state['prompt'] = []
    st.session_state['response'] = []

def hard_reset_session() -> None:
    st.session_state = {states : [] for states in st.session_state}

def create_session_state():
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.0
    if 'token_limit' not in st.session_state:
        st.session_state['token_limit'] = 1024
    if 'top_k' not in st.session_state:
        st.session_state['top_k'] = 1
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.1
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = []
    if 'response' not in st.session_state:
        st.session_state['response'] = []

#running vertex directly here - See above comment
PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID - You will need to set up your GCP project and Region in advance
LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region - You will need to set up your GCP project and Region in advance
vertexai.init(project=PROJECT_ID, location=LOCATION)

#Grabbing the Model - Comment out "Get tuned model" to switch back to base bison model. Bison two is available as option below. There is also an option for the bison2 trained model. It (along with the other tuned model) will expire DEC 21.
@st.cache_resource
def get_model():
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    generation_model = generation_model.get_tuned_model("projects/215149473000/locations/us-central1/models/9086379485103652864")
    #generation_model = TextGenerationModel.from_pretrained("text-bison@002")
    #generation_model = generation_model.get_tuned_model("projects/215149473000/locations/us-central1/models/4907601980857253888")
    return generation_model

def get_text_generation(prompt="",  **parameters):
    generation_model = get_model()
    response = generation_model.predict(prompt=prompt, **parameters
                                        )
    return response.text

#main app function#

st.set_page_config(
    page_title="ASL Team 2 Custom PaLM Contract Summarizer",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This app uses a version of the Vertex PaLM model, fine-tuned on the Atticus Project Contract Understanding Atticus Dataset, to summarize contracts submitted as PDFs."
    }
)

#creating session states
create_session_state()

image = Image.open('./image/FairBanner.png')
st.image(image)
st.title(":red[ASL Team 2] PaLM 2 (Custom-Tuned) Contract Summarizer Demo")

with st.sidebar:
    image = Image.open('./image/FairTab.png')
    st.image(image)
    st.markdown("<h2 style='text-align: center; color: red;'>Output Controls</h2>", unsafe_allow_html=True)

    st.write("Settings:")

    #define the temeperature for the model
    temperature_value = st.slider('Temperature (Output variability) :', 0.0, 1.0, 0.0)
    st.session_state['temperature'] = temperature_value

    #define the temeperature for the model
    token_limit_value = st.slider('Max output length :', 1, 1024, 1024)
    st.session_state['token_limit'] = token_limit_value

    #define the temeperature for the model
    top_k_value = st.slider('Top-K  :', 1,40,1)
    st.session_state['top_k'] = top_k_value

    #define the temeperature for the model
    top_p_value = st.slider('Top-P :', 0.0, 1.0, 0.1)
    st.session_state['top_p'] = top_p_value

    if st.button("Reset Session"):
        reset_session()

# old extract text function - Can remove. This is now handled within the code, after the st.spinner() line
#def extract_text_from_pdf(uploaded_file):
#    text = ""
#    for page in pdfminer.high_level.extract_pages(uploaded_file):
#            text += pdfminer.high_level.extract_text(page)
#    return text

#defining prompt template
prompt_template = """
 
You are a lawyer in a corporate legal department.
Consider the following contract:
<begin contract text>     
```{text}```
<end contract text>
 
FIRST, you must create a one-sentence summary of the contract.
 
SECOND, you must list the following specific information from the contract:
	1) the begin date of the contract
	2) the end date of the contract
	3) the names the human signatories to the contract
THIRD, If this contract is a sales contract, then you must also attentively extract the following additional specific information from the contract:
	1) vendor company name
	2) customer company name
	3) each contract term that pertains to contract cancellation and/or contract renewal (especially automatic renewal)
	4) each contract term that pertains to billing and/or payment
	5) each contract line item (each to include [product description], [product quantity], [product price, with currency if specified], and [charge periodicity (one-time, daily, weekly, monthly, annual, etc.)] for each line item)
	6) each charge not included in the line items (including currency if specified)
 
"""

with st.container():
    st.write("Current Output Settings: ")
    # if st.session_state['temperature'] or st.session_state['debug_mode'] or :
    st.write ("Temperature: ",st.session_state['temperature']," \t \t Max length: ",st.session_state['token_limit']
                ," \t \t Top-K: ",st.session_state['top_k']
                ," \t \t Top-P: ",st.session_state['top_p']
                ," \t \t Debug Model: ",st.session_state['debug_mode'])


    uploaded_file = st.file_uploader("Please upload a contract for summarization. Contracts should be in .pdf format and must be less than 25,000 characters. If your contract is longer, please submit it as parts.", type='pdf')
    #Pass in PDF with upload function
    if uploaded_file:
        with st.spinner('The model is working on your summary...'):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()             
                prompt = prompt_template.format(text=text)
                #Exception handling for length - 
                #Notes from Nick Kelly on exception handling: 
                    #   Change to larger character count if using bison 32 model. 
                    # Also consider a chunking option for future dev.
                    #Try chunking by:
                        #Break text into 25k chunks
                        #Summarize chunk - use general LLM "summarize this" approach
                        #Append summaries of chunks into a new output string
                        #Check new collection of summaries is under 25k
                            #If not, chunk + summarize that
                            #If less than 25k 
                            #Feed into original method below
                if len(prompt) >= 25000:
                    st.markdown("<h6 style='text-align: center; color: red;'>Document exceeds input token limit. Try trimming the document.</h3>", unsafe_allow_html=True)
                #Length exception handling ends
                else:
                    st.markdown("<h3 style='text-align: center; color: red;'>Summary</h3>", unsafe_allow_html=True)
                #getting response from model, pulling prompt from above and session settings defined above    
                    response = get_text_generation(prompt=prompt, temperature = st.session_state['temperature'],
                                    max_output_tokens = st.session_state['token_limit'],
                                    top_p = st.session_state['top_p'],
                                    top_k = st.session_state['top_k'])
                    st.session_state['response'].append(response)
                #writing response out as markdown
                    st.markdown(response)
