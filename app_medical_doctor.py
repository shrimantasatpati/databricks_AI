import streamlit as st
import tempfile
import io
from datetime import datetime
from deep_translator import GoogleTranslator
from langdetect import detect
from gtts import gTTS
from groq import Groq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import base64
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.faiss import FAISS
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from uuid import uuid4
import json
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import base64

# Initialize FAISS index and sentence transformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key="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")

# Initialize Qdrant client
QDRANT_URL = "https://dce73a22-0a9a-4e2b-9f33-1482a96ade76.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "08v2A4ngcvU2hdmySjj6EobniTW4PRqf5U1pAj7eVxgJjApO7Rqzvg"
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Collection names
MEDICAL_DOCS_COLLECTION = "medical_documents"
CONVERSATIONS_COLLECTION = "medical_conversations"

# Create collections if they don't exist
try:
    qdrant_client.get_collection(MEDICAL_DOCS_COLLECTION)
except:
    qdrant_client.create_collection(
        collection_name=MEDICAL_DOCS_COLLECTION,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

try:
    qdrant_client.get_collection(CONVERSATIONS_COLLECTION)
except:
    qdrant_client.create_collection(
        collection_name=CONVERSATIONS_COLLECTION,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

st.set_page_config(
    page_title="Medical Prescription System",
    page_icon="💊",
    layout="wide"
)

# [Previous CSS styles remain exactly the same]
st.markdown("""
    <style>
    .prescription-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        font-family: 'Arial', sans-serif;
    }
    .prescription-header {
        border-bottom: 2px solid #eee;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
    }
    .prescription-header h1 {
        color: #2c3e50;
        font-size: 24px;
        margin-bottom: 8px;
    }
    .prescription-header p {
        color: #7f8c8d;
        margin: 4px 0;
    }
    .patient-details {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    .rx-symbol {
        font-family: serif;
        font-style: italic;
        font-size: 24px;
        color: #2c3e50;
        margin-right: 8px;
    }
    .prescription-content {
        padding: 1rem 0;
    }
    .signature-section {
        margin-top: 2rem;
        text-align: right;
        border-top: 1px solid #eee;
        padding-top: 1rem;
    }
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        margin-top: 1rem;
        display: inline-block;
    }
    .stButton>button {
        width: 100%;
        background-color: #2980b9;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #e3f2fd;
        margin-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def convert_to_english(text):
    try:
        source_lang = detect(text)
    except:
        raise ValueError("Language detection failed.")

    if source_lang == 'en':
        return text, source_lang

    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        raise ValueError(f"Translation failed: {str(e)}")
    return translated_text, source_lang

def translate_to_original(text, target_lang):
    try:
        translated_text = GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        raise ValueError(f"Translation failed: {str(e)}")
    return translated_text

def medical_advice(input_text):
    try:
        prompt = "As a medical professional, provide a detailed response to the following query:\n" + \
                 input_text + "\n\n" + \
                 "Please include:\n" + \
                 "1. General assessment\n" + \
                 "2. Recommended tests or monitoring (if applicable)\n" + \
                 "3. General lifestyle recommendations\n" + \
                 "4. Follow-up suggestions\n\n" + \
                 "Note: Avoid providing specific medication advice."

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.2-3b-preview",
            temperature=0.1,
            max_tokens=2048,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Unable to generate medical response at this time."

def medical_llm(original_query, initial_response):
    prompt = (
        "You are Medical Assistant AI, an expert from the healthcare and biomedical domain. "
        "Your task is to convert the following medical advice into a concise prescription format:\n\n"
        f"Original query: {original_query}\n\n"
        f"Initial response: {initial_response}\n\n"
        "Please format the response in the most minimal way possible:\n"
        "1. Chief Complaints: (maximum 3 most relevant symptoms)\n"
        "2. Clinical Features: (maximum 2-3 key observations)\n"
        "3. Medications: (only essential medications with exact dosage)\n"
        "   Format: Medicine Name Strength, Dose, Duration.\n"
        "4. Investigations: (maximum 2-3 critical tests)\n"
        "5. Advice/Referrals: (maximum 2-3 key points)\n\n"
        "Rules:\n"
        "- Keep each section extremely brief\n"
        "- List only the most essential points\n"
        "- Avoid repetition between sections\n"
        "- For medications, include only specific medicines with complete dosage information\n"
        "- Exclude any generic or obvious advice\n"
        "Do not add any other text or information.\n\n"
    )

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.2-90b-vision-preview",
        temperature=0.1,
        max_tokens=2048,
    )
    
    return chat_completion.choices[0].message.content

# Add new function for doctor chat
def doctor_chat(user_input):
    try:
        # First detect the language of user input
        try:
            detected_lang = detect(user_input)
        except:
            detected_lang = 'en'
            
        # If input is not in English, translate it
        if detected_lang != 'en':
            translated_input = GoogleTranslator(source=detected_lang, target='en').translate(user_input)
        else:
            translated_input = user_input

        prompt = (
            "You are an experienced medical doctor having a conversation with a patient. "
            "For each response:\n"
            "1. Show empathy and understanding of the patient's concerns\n"
            "2. Provide a clear diagnosis if possible\n"
            "3. Recommend specific medications with dosage (use generic names)\n"
            "4. Include any necessary lifestyle modifications or precautions\n"
            "5. Mention when they should come for follow-up\n\n"
            "Keep the tone conversational but professional. Ask follow-up questions if needed for better diagnosis.\n\n"
            f"Patient: {translated_input}"
        )
        
        # Get response from LLM
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.2-3b-preview",
            temperature=0.1,
            max_tokens=1000,
        )
        
        response = chat_completion.choices[0].message.content
        
        # If original input wasn't in English, translate response back
        if detected_lang != 'en':
            try:
                response = GoogleTranslator(source='en', target=detected_lang).translate(response)
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
                # Fall back to English response if translation fails
                pass
                
        return response
        
    except Exception as e:
        st.error(f"Error in doctor's response: {str(e)}")
        return "I apologize, but I'm unable to respond at the moment. Please try again."

def create_pdf_prescription(patient_data, medical_advice):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.grey,
        alignment=1  # Center alignment
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=12,
        fontWeight='bold',
        spaceAfter=6,
        spaceBefore=12
    )
    
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leftIndent=20
    )
    
    rx_style = ParagraphStyle(
        'RxStyle',
        parent=styles['Normal'],
        fontSize=14,
        fontName='Times-Italic',
        spaceBefore=12,
        spaceAfter=12
    )
    
    # Add hospital header
    story.append(Paragraph("AI General Hospital", title_style))
    story.append(Paragraph("Dr. Medical Assistant, MBBS", header_style))
    story.append(Paragraph("Reg. No: REG/123456", header_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d-%b-%Y')}", header_style))
    story.append(Spacer(1, 15))
    
    # Add patient details
    patient_info = f"Name: {patient_data['Name']} | Age: {patient_data['Age']} yrs | Gender: {patient_data['Gender']} | Blood Group: {patient_data['Blood Group']}"
    story.append(Paragraph(patient_info, content_style))
    story.append(Spacer(1, 10))
    
    # Process and format medical advice
    sections = medical_advice.split("\n")
    
    for line in sections:
        line = line.strip()
        # Remove the word "Prescription" if it appears alone
        if line.lower() == "Prescription":
            continue
        if line:
            clean_line = line.replace('**', '')
            if any(section in clean_line for section in ["Chief Complaints:", "Clinical Features:", "Medications:", "Investigations:", "Advice/Referrals:"]):
                story.append(Paragraph(clean_line, section_style))
            else:
                cleaned_line = clean_line.lstrip("- •")
                story.append(Paragraph(f"• {cleaned_line}", content_style))
    
    story.append(Spacer(1, 30))
    
    # Add signature
    signature_style = ParagraphStyle(
        'Signature',
        parent=styles['Normal'],
        fontSize=10,
        alignment=2  # Right alignment
    )
    story.append(Paragraph("Dr. Medical Assistant", signature_style))
    story.append(Paragraph("MBBS", signature_style))
    story.append(Paragraph("Reg. No: REG/123456", signature_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def display_pdf(pdf_bytes):
    """
    Display the PDF in the Streamlit app
    """
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def convert_to_audio(text, lang):
    # try:
    #     tts = gTTS(text=text, lang=lang, slow=False)
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    #         temp_filename = fp.name
    #     tts.save(temp_filename)
    #     return temp_filename
    # except Exception as e:
    #     raise ValueError(f"Audio conversion failed: {str(e)}")
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {str(e)}")
    
def create_medical_report_pdf(text):
    # buffer = io.BytesIO()
    # doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    # styles = getSampleStyleSheet()
    # story = []
    
    # # Custom styles
    # title_style = ParagraphStyle(
    #     'CustomTitle',
    #     parent=styles['Heading1'],
    #     fontSize=16,
    #     spaceAfter=30,
    #     alignment=1  # Center alignment
    # )
    
    # header_style = ParagraphStyle(
    #     'CustomHeader',
    #     parent=styles['Normal'],
    #     fontSize=12,
    #     textColor=colors.grey,
    #     alignment=1  # Center alignment
    # )
    
    # section_style = ParagraphStyle(
    #     'SectionStyle',
    #     parent=styles['Heading2'],
    #     fontSize=12,
    #     fontWeight='bold',
    #     spaceAfter=6,
    #     spaceBefore=12
    # )
    
    # content_style = ParagraphStyle(
    #     'CustomContent',
    #     parent=styles['Normal'],
    #     fontSize=11,
    #     spaceAfter=6,
    #     leftIndent=20
    # )

    # # Add report title
    # story.append(Paragraph("Medical Report", title_style))
    # story.append(Spacer(1, 12))

    # # Parse the medical information string
    # sections = text.split('\n\n')
    # for section in sections:
    #     lines = section.strip().split('\n')
    #     section_title = lines[0]
    #     content = [line.lstrip('- •') for line in lines[1:]]

    #     # Add section
    #     story.append(Paragraph(section_title, section_style))
    #     for item in content:
    #         story.append(Paragraph(f"• {item}", content_style))
    #     story.append(Spacer(1, 12))

    # # Add signature
    # signature_style = ParagraphStyle(
    #     'Signature',
    #     parent=styles['Normal'],
    #     fontSize=10,
    #     alignment=2  # Right alignment
    # )
    # story.append(Spacer(1, 30))
    # story.append(Paragraph("Dr. Medical Assistant", signature_style))
    # story.append(Paragraph("MBBS", signature_style))
    # story.append(Paragraph("Reg. No: REG/123456", signature_style))
    
    # doc.build(story)
    # buffer.seek(0)
    # return buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.grey,
        alignment=1  # Center alignment
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=12,
        fontWeight='bold',
        spaceAfter=6,
        spaceBefore=12
    )
    
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leftIndent=20
    )

    # Add report title
    story.append(Paragraph("Medical Report", title_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d-%b-%Y')}", header_style))
    story.append(Spacer(1, 12))

    # Process the text content
    sections = text.split('\n')
    
    for line in sections:
        line = line.strip()
        # Skip empty lines and standalone "Prescription"
        if not line or line.lower() == "prescription":
            continue
            
        # Clean the line of asterisks and unwanted characters
        clean_line = line.replace('**', '').replace('*', '').strip()
        
        # Check if this is a section header
        if any(section in clean_line for section in [
            "Chief Complaints:", 
            "Clinical Features:", 
            "Medications:", 
            "Investigations:", 
            "Advice/Referrals:",
            "Medical History:",
            "Previous Treatments:",
            "Symptoms:",
            "Current Symptoms:",
            "General Assessment:",
            "Recommended Tests:",
            "Lifestyle Recommendations:",
            "Follow-up Suggestions:"
        ]):
            story.append(Paragraph(clean_line, section_style))
        else:
            # Clean any remaining bullet points or dashes
            cleaned_line = clean_line.lstrip("- •").strip()
            if cleaned_line:  # Only add non-empty lines
                story.append(Paragraph(f"• {cleaned_line}", content_style))

    # Add signature section
    story.append(Spacer(1, 30))
    signature_style = ParagraphStyle(
        'Signature',
        parent=styles['Normal'],
        fontSize=10,
        alignment=2  # Right alignment
    )
    story.append(Paragraph("Dr. Medical Assistant", signature_style))
    story.append(Paragraph("MBBS", signature_style))
    story.append(Paragraph("Reg. No: REG/123456", signature_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer
    
# Function to extract text from documents
def extract_text_from_document(uploaded_file) -> List[Dict]:
    """        
    Returns:
    --------
    List[Dict]
        List of document pages with extracted text
    """
    try:
        # Get file extension
        file_type = uploaded_file.name.lower().split('.')[-1]
        
        # Create temporary file with appropriate extension
        suffix = f'.{file_type}'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        # Choose appropriate loader based on file type
        if file_type == 'pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_type in ['md', 'markdown']:
            loader = TextLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        # Load and extract text
        pages = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return pages
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return []

# Custom embeddings class to wrap SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
def store_document_in_qdrant(documents: List[Dict], doc_type: str) -> bool:
    """Store document text in Qdrant"""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
        )
        splits = text_splitter.split_documents(documents)
        
        # Prepare points for Qdrant
        points = []
        for i, doc in enumerate(splits):
            # Generate embeddings
            embedding = encoder.encode(doc.page_content)
            
            # Create point
            point = models.PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload={
                    'text': doc.page_content,
                    'type': doc_type,
                    'date': datetime.now().isoformat(),
                    'metadata': doc.metadata
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=MEDICAL_DOCS_COLLECTION,
            points=points
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error storing document: {str(e)}")
        return False

def store_conversation_in_qdrant(conversation: str, patient_data: dict) -> bool:
    """Store conversation transcript in Qdrant"""
    try:
        # Generate embedding for the conversation
        embedding = encoder.encode(conversation)
        
        # Create point
        point = models.PointStruct(
            id=str(uuid4()),
            vector=embedding.tolist(),
            payload={
                'conversation': conversation,
                'patient_data': patient_data,
                'date': datetime.now().isoformat()
            }
        )
        
        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=CONVERSATIONS_COLLECTION,
            points=[point]
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error storing conversation: {str(e)}")
        return False

def extract_medical_info_rag(medical_query: str) -> str:
    """Use RAG to extract medical information using Qdrant"""
    try:
        # Generate query embedding
        query_vector = encoder.encode("Medical history").tolist()
        
        # Search for past symptoms
        past_symptoms = qdrant_client.search(
            collection_name=MEDICAL_DOCS_COLLECTION,
            query_vector=query_vector,
            limit=3
        )
        
        # Search for previous treatments
        treatments_query = encoder.encode("Medications").tolist()
        previous_treatments = qdrant_client.search(
            collection_name=MEDICAL_DOCS_COLLECTION,
            query_vector=treatments_query,
            limit=3
        )

        investigation_result = encoder.encode("Past Investigation Results").tolist()
        docs_3 = qdrant_client.search(
            collection_name=MEDICAL_DOCS_COLLECTION,
            query_vector=investigation_result,
            limit=3
        )
        
        # Combine relevant documents
        document_context = "\n\n".join([
            hit.payload['text'] for hit in past_symptoms + previous_treatments + docs_3
        ])
        
        # Create prompt for information extraction
        prompt = f"""
        Based on the following information sources, extract and summarize key medical information:

        Current Query:
        {medical_query}

        Previous Medical Documents:
        {document_context}

        Please extract and structure the following information:
        1. Current Symptoms and Complaints (from {medical_query})
        2. Clinical Features (from {medical_query})
        3. Medical History with all patient details (from {document_context})
        4. All previous Treatments and Medications (from {document_context})
        5. Symptoms (from {document_context})

        NOTE:
        Include only confirmed information from the ```Previous Medical Documents``` and ```Current Query```.
        """

        # Generate structured medical summary using LLM
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="llama-3.2-90b-vision-preview",
            temperature=0.1,
            max_tokens=2048,
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        st.error(f"Error in RAG processing: {str(e)}")
        return "Error processing medical information"
        
def form_medical_question(current_conversation):
    prompt = (
        "You are a medical assistant AI. Based on the following doctor-patient conversation, "
        "please summarize the patient's concerns into a single line query:\n\n"
        f"{current_conversation}\n\n"
        "DO NOT miss out facts, numbers as mentioned in the conversation."
    )
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.2-3b-preview",
        temperature=0.1,
        max_tokens=500,
    )
    
    return chat_completion.choices[0].message.content

def main():
    st.title("🏥 Medical Prescription System")
    
    # Initialize all session states
    if 'prescriptions' not in st.session_state:
        st.session_state.prescriptions = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'full_conversation' not in st.session_state:
        st.session_state.full_conversation = ""
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'stored_documents' not in st.session_state:
        st.session_state.stored_documents = []
    if 'current_prescription_data' not in st.session_state:
        st.session_state.current_prescription_data = {
            'refined_question': None,
            'answer': None,
            'response': None,
            'pdf_buffers': {
                'observation': None,
                'medical_advice': None,
                'prescription': None
            },
            'audio_buffer': None  # New audio buffer in session state
        }
    
    # [Previous sidebar code remains the same]
    with st.sidebar:
        st.header("Settings")
        enable_audio = st.checkbox("Enable Audio Output", value=False)
        
        st.subheader("Upload Medical Documents")
        uploaded_files = st.file_uploader(
            "Upload previous prescriptions or medical reports (PDF)",
            type=['pdf', 'md', 'markdown'],
            accept_multiple_files=True
        )
        
        if st.sidebar.button("Process", use_container_width=True):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        pages = extract_text_from_document(uploaded_file)
                        if pages:
                            success = store_document_in_qdrant(
                                pages,
                                f"Medical Document - {uploaded_file.name}"
                            )
                            if success:
                                st.success(f"Successfully processed {uploaded_file.name}")
                            else:
                                st.error(f"Failed to process {uploaded_file.name}")
                                
    # Three-column layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Patient Information")
        
        patient_data = {
            "Name": st.text_input("Patient Name", key="name"),
            "Age": st.number_input("Age", min_value=0, max_value=150, value=0, key="age"),
            "Gender": st.selectbox("Gender", ["Male", "Female", "Other"], key="gender"),
            "Blood Group": st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"], key="blood_group"),
            "Address": st.text_area("Address", height=50, key="address")
        }

    with col2:
        st.subheader("Chat with Doctor")
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message">👤 {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message bot-message">👨‍⚕️ {message["content"]}</div>',
                        unsafe_allow_html=True
                    )

        user_input = st.text_input("Type your message here...", key="user_input")
        col_send, col_transcript = st.columns([1, 1])
        
        with col_send:
            if st.button("Send", use_container_width=True):
                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    with st.spinner("Doctor is typing..."):
                        response = doctor_chat(user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.full_conversation += f"Patient: {user_input}\nDoctor: {response}\n\n"
                    st.rerun()

        with col_transcript:
            if st.button("Generate, Store Transcript", use_container_width=True):
                if st.session_state.chat_history:
                    if not patient_data["Name"]:
                        st.error("Please fill in patient name before storing conversation.")
                        return
                    
                    success = store_conversation_in_qdrant(
                        st.session_state.full_conversation,
                        patient_data
                    )
                    if success:
                        st.success("Conversation stored successfully!")
                    
                    st.write("Conversation Transcript:")
                    st.text_area(
                        "Transcript",
                        st.session_state.full_conversation,
                        height=200,
                        disabled=True
                    )
    # Modified Preview Column
    with col3:
        st.subheader("Preview & Prescription")
        
        if st.button("Generate Prescription, medical report", use_container_width=True):
            if not patient_data["Name"]:
                st.error("Please fill in patient name.")
                return
            
            if not st.session_state.chat_history:
                st.error("Please have a conversation with the doctor first.")
                return
            
            with st.spinner("Generating prescription..."):
                try:
                    # Generate medical query from conversation
                    medical_query = form_medical_question(st.session_state.full_conversation)
                    st.write(medical_query)
                    
                    # Extract relevant information using RAG
                    st.session_state.current_prescription_data['refined_question'] = extract_medical_info_rag(medical_query)
                    st.write(st.session_state.current_prescription_data['refined_question'])
                    print(st.session_state.current_prescription_data['refined_question'])
                    
                    # Process through translation pipeline
                    english_text, detected_lang = convert_to_english(st.session_state.current_prescription_data['refined_question'])
                    st.session_state.current_prescription_data['answer'] = medical_advice(english_text)
                    st.session_state.current_prescription_data['response'] = medical_llm(english_text, st.session_state.current_prescription_data['answer'])
                    st.write(st.session_state.current_prescription_data['response'])
                    print(st.session_state.current_prescription_data['response'])
                    original_language_text = translate_to_original(st.session_state.current_prescription_data['response'], detected_lang)
                    
                    # Generate PDFs and store in session state
                    st.session_state.current_prescription_data['pdf_buffers']['observation'] = create_medical_report_pdf(st.session_state.current_prescription_data['refined_question'])
                    st.session_state.current_prescription_data['pdf_buffers']['medical_advice'] = create_medical_report_pdf(st.session_state.current_prescription_data['answer'])
                    st.session_state.current_prescription_data['pdf_buffers']['prescription'] = create_pdf_prescription(patient_data, original_language_text)

                    # Generate audio if enabled
                    if enable_audio:
                        with st.spinner("Generating audio..."):
                            # audio_file = convert_to_audio(original_language_text, detected_lang)
                            # st.audio(audio_file)
                            st.session_state.current_prescription_data['audio_buffer'] = convert_to_audio(original_language_text, detected_lang)
                            st.success("Audio generated successfully!")
                    
                    # Save to prescriptions history
                    st.session_state.prescriptions.append({
                        "patient": patient_data.copy(),
                        "advice": original_language_text,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"Error generating prescription: {str(e)}")
                    return
                
        # Display content and download buttons if data exists
        if st.session_state.current_prescription_data['refined_question']:
            st.write("\n\nObservation Report:")
            # st.write(st.session_state.current_prescription_data['refined_question'])
            
            if st.session_state.current_prescription_data['pdf_buffers']['observation']:
                st.download_button(
                    label="📥 Download Observation Report",
                    data=st.session_state.current_prescription_data['pdf_buffers']['observation'],
                    file_name=f"Observation_Report_{patient_data['Name']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            st.write("\n\nMedical Advice:")
            # st.write(st.session_state.current_prescription_data['answer'])
            
            if st.session_state.current_prescription_data['pdf_buffers']['medical_advice']:
                st.download_button(
                    label="📥 Download Medical Advice Report",
                    data=st.session_state.current_prescription_data['pdf_buffers']['medical_advice'],
                    file_name=f"Medical_Advice_Report_{patient_data['Name']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            st.write("\n\nPrescription:")
            # st.write(st.session_state.current_prescription_data['response'])
            
            if st.session_state.current_prescription_data['pdf_buffers']['prescription']:
                # Display PDF preview
                display_pdf(st.session_state.current_prescription_data['pdf_buffers']['prescription'].getvalue())
                
                # Download button
                st.download_button(
                    label="📥 Download Prescription",
                    data=st.session_state.current_prescription_data['pdf_buffers']['prescription'],
                    file_name=f"prescription_{patient_data['Name']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            # Display audio player if audio is available
            if enable_audio and st.session_state.current_prescription_data['audio_buffer']:
                st.audio(st.session_state.current_prescription_data['audio_buffer'])
        
        # Recent prescriptions display
        if st.session_state.prescriptions:
            with st.expander("📋 Recent Prescriptions"):
                for prescription in reversed(st.session_state.prescriptions[-5:]):
                    st.write(f"**Patient:** {prescription['patient']['Name']}")
                    st.write(f"**Date:** {prescription['date']}")
                    st.write("\n**Medical Advice:**")
                    st.write(prescription['advice'])
                    st.divider()

if __name__ == "__main__":
    main()