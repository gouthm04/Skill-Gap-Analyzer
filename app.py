"""
SkillBridge: Career Growth Assistant (Hackathon Edition)
A Streamlit application that uses Groq AI to analyze resumes
and identify skill gaps against a specific Job Description.
"""

import streamlit as st
from groq import Groq
import os
import sys
import traceback
import json
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in environment variables! Please set it in your .env file.")
    st.stop()

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Groq API: {str(e)}")
    st.stop()


def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a provided PDF file object."""
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def analyze_resume(resume_text: str, job_role: str, job_description: str) -> dict:
    """
    Analyze resume using Groq AI.
    """
    
    jd_text = job_description if job_description.strip() else "(Not provided. Please infer the standard industry requirements based purely on the Target Job Role.)"
    
    prompt = f"""
You are an expert AI Career Copilot.
Your task is to deeply analyze a candidate's resume against a target job role and its specific job description.

Resume:
{resume_text}

Target Job Role:
{job_role}

Job Description Requirements:
{jd_text}

You must return your analysis STRICTLY in JSON format matching the exact structure below. Do not include markdown code blocks, just the raw JSON object.

{{
  "match_score": 72,
  "matched_skills": ["React", "Node.js"],
  "missing_skills": ["Docker", "AWS"],
  "ats_score": 78,
  "ats_analysis": {{
    "missing_keywords": ["Kubernetes", "CI/CD"],
    "formatting_issues": ["Long paragraphs", "Weak action verbs"]
  }},
  "learning_roadmap": [
    {{
      "week": "Week 1",
      "topic": "Learn Docker Basics",
      "resources": ["Docker Official Docs", "Docker for Beginners on YouTube"],
      "practice_task": "Containerize a simple Node.js app",
      "estimated_time": "8 hours"
    }}
  ],
  "project_recommendations": [
    {{
      "name": "Microservices E-commerce",
      "difficulty": "Intermediate",
      "tech_stack": ["Node.js", "Docker", "MongoDB"],
      "description": "Demonstrates containerization and backend architecture."
    }}
  ],
  "resume_improvements": [
    {{
      "original": "Worked on a web project",
      "suggested": "Developed a responsive web application serving 500+ users, optimizing load time by 30%."
    }}
  ]
}}
"""
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional AI career assistant that strictly outputs JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )
        
        result_text = completion.choices[0].message.content
        return json.loads(result_text)
    
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON response from AI: {str(e)}\n\nResponse was:\n{result_text}"}
    except Exception as e:
        return {"error": f"Error analyzing resume: {str(e)}"}

def plot_skill_match(matched, missing):
    """Generates a pie chart for skill matching"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if not matched and not missing:
        # Edge case: no skills found
        ax.pie([1], labels=['No Data'], colors=['#cccccc'])
        return fig
        
    labels = ['Matched Skills', 'Missing Skills']
    sizes = [len(matched), len(missing)]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)  
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90, textprops={'color':"w", 'weight':'bold'})
    ax.axis('equal')  
    
    fig.patch.set_alpha(0.0)
    return fig

def main():
    st.set_page_config(page_title="SkillBridge | Career Copilot", page_icon="🌉", layout="wide")
    
    st.title("🌉 SkillBridge: Career Growth Assistant")
    st.markdown("Go beyond a basic resume score. Upload your resume and the target job description to get a personalized action plan.")
    
    st.divider()
    
    # Custom CSS to make tabs look like buttons with gaps
    st.markdown("""
    <style>
        /* Add gap between tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
        }
        /* Style individual tabs as buttons */
        .stTabs [data-baseweb="tab"] {
            background-color: #2b2b36;
            border-radius: 8px;
            padding: 8px 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* Style the active tab */
        .stTabs [aria-selected="true"] {
            background-color: #FF4B4B !important;
            color: white !important;
        }
        /* Remove the default underline */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ Your Resume")
        input_method = st.radio("Choose input method:", ["Upload PDF", "Paste Text"], horizontal=True)
        
        resume_text = ""
        if input_method == "Upload PDF":
            uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
            if uploaded_file is not None:
                resume_text = extract_text_from_pdf(uploaded_file)
                if resume_text.startswith("Error"):
                    st.error(resume_text)
                    resume_text = ""
                elif not resume_text.strip():
                    st.error("⚠️ No text could be extracted from this PDF. It might be a scanned image or use an unsupported font encoding. Please try the 'Paste Text' option instead.")
                    resume_text = ""
                else:
                    st.success("PDF uploaded and parsed successfully!")
                    with st.expander("View Parsed Text"):
                        st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        else:
            resume_text = st.text_area("Paste your resume here", height=250, placeholder="Paste your complete resume text here...")
            
    with col2:
        st.subheader("2️⃣ The Job")
        job_role = st.text_input("Target Job Role", placeholder="e.g., Senior Frontend Engineer")
        job_description = st.text_area("Job Description (Optional - Paste the posting)", height=250, placeholder="Optional: Paste the full job description requirements here for a better analysis...")
        
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("🚀 Analyze & Generate Career Plan", type="primary", use_container_width=True)
        
    st.divider()
    
    if analyze_button:
        if not resume_text.strip():
            st.warning("⚠️ Please provide your resume.")
            return
        if not job_role.strip():
            st.warning("⚠️ Please enter the target job role.")
            return
            
        with st.spinner("🤖 SkillBridge is analyzing your profile against the job requirements..."):
            data = analyze_resume(resume_text, job_role, job_description)
            
        if "error" in data:
            st.error(data["error"])
        else:
            # Display Results Dashboard
            st.header("📊 Your SkillBridge Dashboard")
            
            # Create Tabs for a cleaner Dashboard UI
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Overview & ATS", 
                "🗺️ Learning Roadmap", 
                "💡 Portfolio Projects", 
                "✍️ Resume Rewrites"
            ])
            
            # --- TAB 1: OVERVIEW & ATS ---
            with tab1:
                m1, m2, m3 = st.columns([1, 1, 1.5])
                
                with m1:
                    st.metric("🎯 Match Score", f"{data.get('match_score', 0)}%")
                    st.progress(max(0.0, min(1.0, data.get('match_score', 0) / 100.0)))
                    
                with m2:
                    st.metric("📄 ATS Score", f"{data.get('ats_score', 0)}/100")
                    st.progress(max(0.0, min(1.0, data.get('ats_score', 0) / 100.0)))
                    
                with m3:
                    matched = data.get("matched_skills", [])
                    missing = data.get("missing_skills", [])
                    if matched or missing:
                        fig = plot_skill_match(matched, missing)
                        st.pyplot(fig, transparent=True)
                
                st.divider()
                
                col_sk1, col_sk2, col_ats = st.columns(3)
                
                with col_sk1:
                    st.subheader("✅ Matched Skills")
                    for skill in matched:
                        st.markdown(f"- {skill}")
                        
                with col_sk2:
                    st.subheader("❌ Missing Skills")
                    for skill in missing:
                        st.markdown(f"- {skill}")
                        
                with col_ats:
                    st.subheader("🤖 ATS Analysis")
                    ats_data = data.get("ats_analysis", {})
                    
                    st.markdown("**Missing Keywords:**")
                    for kw in ats_data.get("missing_keywords", []):
                        st.markdown(f"- ✗ {kw}")
                        
                    st.markdown("**Formatting Issues:**")
                    for issue in ats_data.get("formatting_issues", []):
                        st.markdown(f"- • {issue}")
                        
            # --- TAB 2: LEARNING ROADMAP ---
            with tab2:
                st.subheader("🗺️ Personalized Learning Roadmap")
                st.markdown("Your week-by-week action plan to bridge the skill gap.")
                
                for week in data.get("learning_roadmap", []):
                    with st.expander(f"📅 {week.get('week', '')} - {week.get('topic', '')} (⏱️ {week.get('estimated_time', 'N/A')})", expanded=True):
                        
                        st.markdown("**📚 Resources:**")
                        for res in week.get("resources", []):
                            st.markdown(f"- {res}")
                            
                        st.markdown(f"**💻 Practice Task:**\n{week.get('practice_task', '')}")
                        
            # --- TAB 3: PROJECTS ---
            with tab3:
                st.subheader("💡 Actionable Portfolio Projects")
                st.markdown("Build these specific projects to cover your missing skills.")
                
                # Display projects in a grid
                proj_cols = st.columns(len(data.get("project_recommendations", [])) or 1)
                
                for idx, proj in enumerate(data.get("project_recommendations", [])):
                    with proj_cols[idx % len(proj_cols)]:
                        st.info(f"### {proj.get('name', 'Project')}")
                        st.caption(f"Difficulty: {proj.get('difficulty', 'Unknown')}")
                        
                        tech_stack = ", ".join(proj.get("tech_stack", []))
                        st.markdown(f"**Uses:** {tech_stack}")
                        st.markdown(proj.get("description", ""))
                
            # --- TAB 4: RESUME REWRITES ---
            with tab4:
                st.subheader("✍️ Resume Rewrites")
                st.markdown("Enhance your bullet points to pass ATS systems.")
                
                for imp in data.get("resume_improvements", []):
                    st.markdown("**❌ Current:**")
                    st.code(imp.get('original', ''), language="text")
                    st.markdown("**✅ Improved:**")
                    st.code(imp.get('suggested', ''), language="text")
                    st.divider()

if __name__ == "__main__":
    main()
