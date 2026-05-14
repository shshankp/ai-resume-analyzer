# SmartProductAgent

SmartProductAgent is an AI-powered shopping assistant and recommendation platform that helps users discover products, compare prices, monitor price changes, and receive intelligent recommendations using AI-driven workflows and multi-agent systems.

---

## Features

- AI-powered product recommendations
- Smart product search and filtering
- Price tracking and monitoring
- Real-time product analytics
- Intelligent alert and notification system
- Multi-agent workflow architecture
- Personalized shopping suggestions
- API integration support
- Interactive and responsive dashboard UI

---

## Tech Stack

### Backend
- Python
- Flask / FastAPI

### Frontend
- HTML
- CSS
- JavaScript

### AI & Data
- Gemini API
- Recommendation Engine
- Multi-Agent System
- REST APIs

---

## Project Structure

```txt
SmartProductAgent/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── static/
├── templates/
├── uploads/
│
├── agents/
│   ├── recommendation_agent.py
│   ├── product_search_agent.py
│   ├── analytics_agent.py
│   └── alert_agent.py
│
├── recommendation/
│
├── src/
│
├── utils/
│   ├── helpers.py
│   ├── prompts.py
│   └── api_utils.py
│
└── data/
```

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/SmartProductAgent.git
cd SmartProductAgent
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

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

## Gemini API Notes

Set your Gemini API key in environment variables.

### Windows

```bash
set GEMINI_API_KEY=your_api_key
```

### Mac/Linux

```bash
export GEMINI_API_KEY=your_api_key
```

If the API key is unavailable, the application will continue running using fallback recommendation workflows.

---

## Run Project

```bash
python app.py
```

or

```bash
python main.py
```

---

## Future Improvements

- Voice-enabled shopping assistant
- Advanced AI personalization
- Product sentiment analysis
- Wishlist synchronization
- E-commerce integrations
- Real-time deal detection
- Mobile application support

---

## License

This project is licensed under the MIT License

---

## Author

**Shshank**

GitHub: https://github.com/shshankp

LinkedIn: https://linkedin.com/in/shshankp

---

