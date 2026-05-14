# AI Job Search & Resume Improvement Agent

An AI-powered career assistant that analyzes resumes, extracts skills, matches users with relevant jobs, and generates intelligent recommendations using LLM-driven workflows.

---

## Features

- Resume analysis and skill extraction
- AI-powered job matching
- Resume improvement suggestions
- Interview question generation
- Recommendation agent for career guidance
- Job filtering and search support
- Gemini API integration
- Streamlit-based interactive UI

---

## Tech Stack

### Backend
- Python

### Frontend
- Streamlit

### AI & APIs
- Gemini API
- LLM-based Recommendation System
- Resume Parsing
- JSON-based Job Dataset

---

## Project Structure

```txt
project/
│
├── app.py
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── job_finder.py
│   ├── recommendation_agent.py
│   ├── resume_analyzer.py
│   └── __init__.py
│
├── services/
│   ├── env_config.py
│   ├── llm_client.py
│   ├── resume_text.py
│   └── __init__.py
│
├── models/
│   ├── schemas.py
│   └── __init__.py
│
├── data/
│   └── sample_jobs.json
│
└── README.md
```

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-job-search-agent.git
cd ai-job-search-agent
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key
```

You can copy from:

```bash
cp .env.example .env
```

---

## Run the Project

```bash
streamlit run app.py
```

---

## AI Agents

### Resume Analyzer Agent
- Extracts skills
- Parses resume text
- Identifies experience and technologies

### Job Finder Agent
- Matches jobs based on skills
- Filters relevant opportunities
- Uses intelligent scoring logic

### Recommendation Agent
- Suggests improvements
- Generates career guidance
- Creates interview questions

---

## Future Improvements

- LinkedIn job integration
- Real-time job APIs
- ATS score calculation
- Resume PDF export
- Cover letter generation
- Multi-language support
- Authentication system

---

## License

This project is licensed under the MIT License.

---

## Author

**Shshank**

GitHub: https://github.com/shshankp

LinkedIn: https://linkedin.com/in/shshankp
