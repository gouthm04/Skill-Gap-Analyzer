"""
SkillBridge: Career Growth Assistant (Hackathon Edition)
A Streamlit application that uses Groq AI to analyze resumes
and identify skill gaps against a specific Job Description.
"""

import streamlit as st
from groq import Groq
import os
import json
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables! Please set it in your .env file.")
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
  "hiring_probability": 78,
  "match_score": 72,
  "matched_skills": ["React", "Node.js"],
  "missing_skills": ["Docker", "AWS"],
  "ats_score": 78,
  "ats_analysis": {{
    "missing_keywords": ["Kubernetes", "CI/CD"],
    "formatting_issues": ["Long paragraphs", "Missing Summary"],
    "strengths": ["Action verbs", "Quantified achievements"]
  }},
  "learning_roadmap": [
    {{
      "week": "Week 1",
      "topic": "Learn Docker Basics",
      "estimated_hours": "10 hours",
      "difficulty": "2/5",
      "resources": ["Docker Official Docs", "Docker for Beginners on YouTube"],
      "practice_task": "Containerize a simple Node.js app"
    }}
  ],
  "project_recommendations": [
    {{
      "name": "Inventory Management",
      "difficulty": "Intermediate",
      "tech_stack": ["React", "Node.js", "MongoDB"],
      "build_time": "2 weeks",
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


def main():
    st.set_page_config(page_title="SkillBridge", layout="wide", page_icon="🚀")
    
    # Clean header without aggressive CSS
    st.title("SkillBridge: AI Career Growth Assistant")
    st.markdown("Analyze your resume. Discover skill gaps. Get a personalized plan to achieve your dream role.")
    st.divider()
    
    # Input Area using native container borders
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("1. Your Resume")
            st.caption("Upload your current resume (PDF)")
            
            if 'input_mode' not in st.session_state:
                st.session_state.input_mode = 'pdf'
                
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Upload PDF", type="primary" if st.session_state.input_mode == 'pdf' else "secondary", use_container_width=True):
                    st.session_state.input_mode = 'pdf'
                    st.rerun()
            with b2:
                if st.button("Paste Text", type="primary" if st.session_state.input_mode == 'text' else "secondary", use_container_width=True):
                    st.session_state.input_mode = 'text'
                    st.rerun()
            
            pdf_text = ""
            pasted_text = ""
            
            if st.session_state.input_mode == 'pdf':
                uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
                if uploaded_file is not None:
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if not pdf_text.strip():
                        st.error("No text could be extracted.")
                        pdf_text = ""
                    elif pdf_text.startswith("Error"):
                        st.error(pdf_text)
                        pdf_text = ""
                    else:
                        st.success("PDF uploaded!")
            else:
                pasted_text = st.text_area("Paste your resume here", height=200, label_visibility="collapsed")
                
            resume_text = pasted_text if pasted_text.strip() else pdf_text
                
    with col2:
        with st.container(border=True):
            st.subheader("2. The Job")
            st.caption("Enter the target role and job description")
            
            st.markdown("**Target Job Role**")
            job_role = st.text_input("Target Job Role", placeholder="e.g., Senior Frontend Engineer", label_visibility="collapsed")
            
            st.markdown("**Job Description** (Recommended)")
            job_description = st.text_area("Job Description", height=130, placeholder="Paste the actual job posting from LinkedIn, Indeed, or a company careers page here...", label_visibility="collapsed")
            
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        analyze_button = st.button("Analyze Resume & Generate My Career Plan", type="primary", use_container_width=True)
    
    if analyze_button:
        if not resume_text.strip():
            st.warning("Please provide your resume.")
        elif not job_role.strip():
            st.warning("Please enter the target job role.")
        else:
            with st.spinner("SkillBridge is analyzing your profile against the job requirements..."):
                data = analyze_resume(resume_text, job_role, job_description)
                
            if "error" in data:
                st.error(data["error"])
            else:
                st.session_state.analysis_data = data
                st.session_state.show_dashboard = True
                
    if not st.session_state.get('show_dashboard', False):
        st.divider()
        st.subheader("What you'll get")
        
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            with st.container(border=True):
                st.info("🛡️ ATS Analysis")
                st.caption("Get your ATS score and optimize for applicant tracking systems")
        with f2:
            with st.container(border=True):
                st.success("🎯 Skill Gap Detection")
                st.caption("Identify missing skills and prioritize what to learn")
        with f3:
            with st.container(border=True):
                st.warning("🗺️ Learning Roadmap")
                st.caption("Personalized week-by-week plan to bridge your skill gaps")
        with f4:
            with st.container(border=True):
                st.error("💼 Portfolio Projects")
                st.caption("Curated project ideas to boost your practical experience")
        with f5:
            with st.container(border=True):
                st.info("📝 Resume Rewrites")
                st.caption("AI-powered suggestions to make your resume stand out")
                
    else:
        data = st.session_state.analysis_data
        st.divider()
        st.header("Your SkillBridge Dashboard")
            
        if 'dash_tab' not in st.session_state:
            st.session_state.dash_tab = 'Overview & ATS'
            
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            if st.button("Overview & ATS", type="primary" if st.session_state.dash_tab == 'Overview & ATS' else "secondary", use_container_width=True):
                st.session_state.dash_tab = 'Overview & ATS'
                st.rerun()
        with t2:
            if st.button("Learning Roadmap", type="primary" if st.session_state.dash_tab == 'Learning Roadmap' else "secondary", use_container_width=True):
                st.session_state.dash_tab = 'Learning Roadmap'
                st.rerun()
        with t3:
            if st.button("Portfolio Projects", type="primary" if st.session_state.dash_tab == 'Portfolio Projects' else "secondary", use_container_width=True):
                st.session_state.dash_tab = 'Portfolio Projects'
                st.rerun()
        with t4:
            if st.button("Resume Rewrites", type="primary" if st.session_state.dash_tab == 'Resume Rewrites' else "secondary", use_container_width=True):
                st.session_state.dash_tab = 'Resume Rewrites'
                st.rerun()
        
        # --- TAB 1: OVERVIEW & ATS ---
        if st.session_state.dash_tab == 'Overview & ATS':
            st.subheader("Executive Summary")
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric("Match Score", f"{data.get('match_score', 0)}%")
                
            with m2:
                st.metric("ATS Score", f"{data.get('ats_score', 0)}/100")
                
            with m3:
                st.metric("Hiring Probability", f"{data.get('hiring_probability', 0)}%")
            
            st.divider()
            
            col_sk1, col_sk2, col_ats = st.columns([1, 1, 1.2])
            
            with col_sk1:
                with st.container(border=True):
                    matched = data.get("matched_skills", [])
                    st.subheader(f"Matched Skills ({len(matched)})")
                    if matched:
                        # Using native progress 
                        st.progress(1.0)
                    for skill in matched:
                        st.markdown(f"- {skill}")
                    
            with col_sk2:
                with st.container(border=True):
                    missing = data.get("missing_skills", [])
                    st.subheader(f"Missing Skills ({len(missing)})")
                    if missing:
                        st.progress(0.4)
                    for skill in missing:
                        st.markdown(f"- {skill}")
                    
            with col_ats:
                with st.container(border=True):
                    st.subheader("ATS Analysis")
                    ats_data = data.get("ats_analysis", {})
                    
                    st.markdown("**Strengths:**")
                    for s in ats_data.get("strengths", []):
                        st.markdown(f"- ✓ {s}")
                        
                    st.markdown("**Missing Keywords:**")
                    for kw in ats_data.get("missing_keywords", []):
                        st.markdown(f"- ✗ {kw}")
                        
                    st.markdown("**Formatting Issues:**")
                    for issue in ats_data.get("formatting_issues", []):
                        st.markdown(f"- ⚠ {issue}")
                    
        # --- TAB 2: LEARNING ROADMAP ---
        if st.session_state.dash_tab == 'Learning Roadmap':
            st.subheader("Personalized Learning Roadmap")
            st.markdown("Your week-by-week action plan to bridge the skill gap.")
            
            for week in data.get("learning_roadmap", []):
                with st.expander(f"{week.get('week', '')} - {week.get('topic', '')}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Estimated Time:** {week.get('estimated_hours', 'N/A')}")
                    with col2:
                        st.markdown(f"**Difficulty:** {week.get('difficulty', 'N/A')}")
                        
                    st.markdown("**Resources:**")
                    for res in week.get("resources", []):
                        st.markdown(f"- {res}")
                        
                    st.markdown(f"**Practice Task:**\n{week.get('practice_task', '')}")
                    
        # --- TAB 3: PROJECTS ---
        if st.session_state.dash_tab == 'Portfolio Projects':
            st.subheader("Actionable Portfolio Projects")
            st.markdown("Build these specific projects to cover your missing skills.")
            
            for proj in data.get("project_recommendations", []):
                with st.container(border=True):
                    st.markdown(f"### {proj.get('name', 'Project')}")
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.markdown(f"**Difficulty:** {proj.get('difficulty', 'Unknown')}")
                    with p2:
                        tech_stack = ", ".join(proj.get("tech_stack", []))
                        st.markdown(f"**Tech Stack:** {tech_stack}")
                    with p3:
                        st.markdown(f"**Build Time:** {proj.get('build_time', 'Unknown')}")
                    
                    st.markdown(f"*{proj.get('description', '')}*")
            
        # --- TAB 4: RESUME REWRITES ---
        if st.session_state.dash_tab == 'Resume Rewrites':
            st.subheader("Resume Rewrites")
            st.markdown("Enhance your bullet points to pass ATS systems.")
            
            for imp in data.get("resume_improvements", []):
                with st.container(border=True):
                    st.markdown("**Current:**")
                    st.code(imp.get('original', ''), language="text")
                    st.markdown("**Improved:**")
                    st.code(imp.get('suggested', ''), language="text")

if __name__ == "__main__":
    main()
