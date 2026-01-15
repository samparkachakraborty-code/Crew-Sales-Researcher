from crewai import Agent , Task , Crew , LLM , Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import os 


#load of llm 
os.environ["OPENAI_API_KEY"] = "NA"
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    api_key="ollama",
    config = {
        "max_tokens": 4096,
        "temperature" : 0.2,
        "top_p" : 0.9
    }
)

#searchtool setup
shared_search_memory = {}
ddg_searchtool = DuckDuckGoSearchRun()
@tool("internet_search_tool")
def internet_search_tool(query : str):
    """Search and store results in shared memory."""
    query = query.lower().strip()
    if query in shared_search_memory :
        return f"[FROM MEMORY]\n{shared_search_memory[query]}"
    result = ddg_searchtool.run(query)
    shared_search_memory[query] = result 
    return result 


#user_input

company = input("Enter a company name : ")

#agentsetup : 

research_agent = Agent(
    role = "Company Research Analyst",
    goal = f"""Research at {company} using verified internet search results of last 6-12 months and :
           - Identify the pain-points
           - The challenges they are facing
           
           Rules : 
           - If the information is unavailabel , explicitly write : "Not-Found"
           - Don't create or guess fake informations
           - Always use verified internet search results
           - Use shared research memory before performing new searches.
           - Do not repeat searches already stored in memory.

           """,
    backstory = "You are an expert business analyst , who gathers and generates relevent insights , your provided all informations must be accurate",
    tools = [internet_search_tool],
    llm= local_llm,
    max_iter = 3,
    max_execution_time = 60,
    verbose = True
)

decision_maker_agent = Agent(
    role = "Decision Makers Identifier",
    goal = f"""Identify only verified key decision makers at {company} , and focus on :
           - The full names of the decision makers. 
           - The Job-Titles of them
           - VP of sales and CTO
           - Use shared research memory before performing new searches.
           - Do not repeat searches already stored in memory.

           """,
    backstory = "You are an expert at identifying Leadership roles and Desicion Makers relevant to B2B sales",
    tools = [internet_search_tool],
    llm= local_llm,
    verbose = True
)

email_writer_agent = Agent(
    role = "Sales Email Copywriter (premium version of B2B Copywriter)",
    goal = """Write a personalised , pain-point-focused sales email using - 
           - Research task output
           - Decision maker task output
           - Use shared research memory before performing new searches.
           - Do not repeat searches already stored in memory.

           """,
    backstory = "You are an expert at top-performing B2B sales copywriter , who writes short personalised - human sounding emails with key pain-points",
    llm= local_llm,
    verbose = True
)

#tasksetup

research_task = Task(
    description=f"""
    Research {company} and provide,
    - The recent news atleast 6-12 months ago
    - Business focus
    - Their key pain points and challenges

    Strict Rules : 
    - Provided every information must supported by internet search result and must have a source
    - If information is unavailable, explicitly write: "Not-Found"
    - Don't guess or assume or infer
    """,
    expected_output=""" A well structured output such as : 
                    1. Recent News - 
                    News :
                    Source :
                    2. Business Focus - 
                    Point :
                    Source : 
                    3. Pain-points and Challenges -
                    Pain-points :
                    Source : 
                    """,
    agent=research_agent
)

decision_maker_task = Task(
    description=f"""
    Identify leadership and the decision maker to {company}
    Focus on : 
    1. VP of sales
    2. CTO
    3. Head of Business Marketing

    Strict Rules : 
    1. Use only verified resources
    2. Don't provide fake data
    3. If any information or data is not available , explicitly write : "Not-Found"
    4. Don't invent name or roles
    """,
    expected_output="""A well structured Table  :
                    [Name | Job-Title | Department | Source | Confidence (High , Medium , Low)]       
    """,
    agent=decision_maker_agent,
    context=[research_task]
)

email_writer_task = Task(
    description=f"""
    Write a cold email to {company} , with help of research and decision maker info
    The email should including points such as : 
    1. The recent news
    2. Their 2-3 pain points , atleast 1 major pain-point should be highlighted in the email
    3. Approach for a cold call or a meeting regarding the solutions of the pain points 

    "Critical Rule: 
    - The email must not contradict the research found. 
    - If the research says the company is profitable, do not list 'loss-making' as a pain point. "

    The email should sound  : 
    1. Concise
    2. Professional 
    3. Human and natural tone
    """,
    expected_output="A short and personalized cold email draft",
    agent=email_writer_agent,
    human_input=True
)

#crew setup 


crew = Crew(
    agents=[research_agent , decision_maker_agent , email_writer_agent],
    tasks=[research_task , decision_maker_task , email_writer_task],
    process = Process.sequential,
    verbose=True
)

result = crew.kickoff()

print("__________Final Output__________")
print("\n")
print(result)
