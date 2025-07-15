import streamlit as st
import os
from crewai.tools import tool
from exa_py import Exa
from crewai import LLM, Agent, Task, Crew

# Set up page configuration
st.set_page_config(
    page_title="LinkedIn Prospect Finder",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 LinkedIn Prospect Finder")
st.markdown("Find key prospects at target companies using AI-powered search")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# API Key setup using st.secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    exa_api_key = st.secrets["EXA_API_KEY"]
    
    # Set environment variables
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['EXA_API_KEY'] = exa_api_key
    
except KeyError as e:
    st.error(f"Missing API key in secrets: {e}")
    st.info("Please add your API keys to Streamlit secrets:")
    st.code("""
    # .streamlit/secrets.toml
    OPENAI_API_KEY = "your-openai-api-key"
    EXA_API_KEY = "your-exa-api-key"
    """)
    st.stop()

# Define the Exa search tool
@tool("Exa search and get contents")
def search_and_get_contents_tool(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""
    
    exa = Exa(exa_api_key)
    
    response = exa.search_and_contents(
        question,
        type="neural",
        num_results=3,
        highlights=True
    )
    
    parsedResult = ''.join([
        f'<Title id={idx}>{eachResult.title}</Title>'
        f'<URL id={idx}>{eachResult.url}</URL>'
        f'<Highlight id={idx}>{"".join(eachResult.highlights)}</Highlight>'
        for (idx, eachResult) in enumerate(response.results)
    ])
    
    return parsedResult

# Initialize LLM
@st.cache_resource
def initialize_llm():
    return LLM(model='openai/gpt-4.1-mini-2025-04-14', temperature=0)

# Create the CrewAI setup
@st.cache_resource
def setup_crew():
    llm_openai_leads = initialize_llm()
    
    LinkedIn_finder_agent = Agent(
        role="LinkedIn Sales Intelligence Analyst",
        goal="Find 5 key prospects at target companies and retrieve their LinkedIn profile URLs",
        backstory="""Expert at finding and identifying key prospects within target companies using LinkedIn search. Specialized in company-based prospecting and finding the right decision makers within target organizations.""",
        tools=[search_and_get_contents_tool],
        llm=llm_openai_leads,
        verbose=True,
        memory=True
    )
    
    LinkedIn_finder_task = Task(
        description="""
        Research the target company {company} and find 5 key prospects within that organization.

        REQUIREMENTS:
        - Search for employees at the target company using LinkedIn
        - Identify 5 key prospects in relevant roles (decision makers, influencers, end users)
        - Focus on roles like: VP/Director of IT, CTO, Engineering Manager, Operations Manager, etc.
        - Prioritize prospects likely to have budget authority or influence over purchasing decisions
        - Extract their LinkedIn profile URLs

        Input: Company name
        """,
        expected_output="""
        **PROSPECT LIST FOR: [Company Name]**

        | Name | Job Title | Department | LinkedIn Profile URL |
        |------|-----------|------------|---------------------|
        | John Smith | VP of Engineering | Engineering | https://linkedin.com/in/johnsmith |
        | Jane Doe | CTO | Technology | https://linkedin.com/in/janedoe |
        | Michael Johnson | Director of Operations | Operations | https://linkedin.com/in/michaeljohnson |
        | Sarah Wilson | Head of IT | Information Technology | https://linkedin.com/in/sarahwilson |
        | David Brown | Engineering Manager | Engineering | https://linkedin.com/in/davidbrown |

        **Total Prospects Found: 5**
        """,
        agent=LinkedIn_finder_agent,
        output_file="linkedin.md"
    )
    
    crew = Crew(
        agents=[LinkedIn_finder_agent],
        tasks=[LinkedIn_finder_task]
    )
    
    return crew

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Company Search")
    company_name = st.text_input(
        "Enter Company Name:",
        placeholder="e.g., Microsoft, Google, Apple",
        help="Enter the name of the company you want to find prospects for"
    )
    
    search_button = st.button(
        "🔍 Find Prospects",
        type="primary",
        disabled=not company_name or st.session_state.is_running
    )
    
    if st.session_state.is_running:
        st.info("🔄 Searching for prospects... This may take a few minutes.")
        
    # Clear results button
    if st.session_state.results:
        if st.button("Clear Results"):
            st.session_state.results = None
            st.rerun()

with col2:
    st.subheader("How it works")
    st.markdown("""
    1. **Enter Company Name**: Type the target company name
    2. **AI Search**: The system searches for key employees on LinkedIn
    3. **Prospect Analysis**: AI identifies decision makers and influencers
    4. **Results**: Get a formatted list with LinkedIn profiles
    
    **Target Roles:**
    - VP/Director of IT
    - CTO, Engineering Managers
    - Operations Managers
    - Budget decision makers
    """)

# Process search
if search_button and company_name:
    st.session_state.is_running = True
    st.rerun()

if st.session_state.is_running:
    try:
        with st.spinner(f"Searching for prospects at {company_name}..."):
            crew = setup_crew()
            result = crew.kickoff({"company": company_name})
            
            st.session_state.results = result
            st.session_state.is_running = False
            st.rerun()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.is_running = False
        st.rerun()

# Display results
if st.session_state.results:
    st.subheader("🎯 Prospect Results")
    
    # Display the markdown results
    st.markdown(str(st.session_state.results))
    
    # Add download button for results
    st.download_button(
        label="📥 Download Results as Markdown",
        data=str(st.session_state.results),
        file_name=f"linkedin_prospects_{company_name.lower().replace(' ', '_')}.md",
        mime="text/markdown"
    )
    
    # Additional information
    st.success("✅ Prospect search completed successfully!")
    
    with st.expander("ℹ️ Next Steps"):
        st.markdown("""
        **Recommended Actions:**
        1. **Review Profiles**: Click on LinkedIn URLs to view full profiles
        2. **Personalize Outreach**: Research each prospect's background
        3. **Track Engagement**: Monitor response rates and interactions
        4. **Follow Up**: Schedule appropriate follow-up sequences
        
        **Best Practices:**
        - Personalize each message based on their role and company
        - Reference recent company news or achievements
        - Provide clear value proposition
        - Keep initial messages brief and professional
        """)

# Sidebar with additional info
with st.sidebar:
    st.header("📊 Tool Information")
    st.markdown("""
    **Powered by:**
    - CrewAI for multi-agent coordination
    - Exa for semantic search
    - OpenAI GPT-4 for intelligence
    
    **Features:**
    - AI-powered prospect identification
    - LinkedIn profile extraction
    - Decision maker prioritization
    - Formatted results export
    """)
    
    st.header("🔧 Requirements")
    st.markdown("""
    **API Keys Needed:**
    - OpenAI API Key
    - Exa API Key
    
    Add these to your Streamlit secrets configuration.
    """)
    
    st.header("📝 Tips")
    st.markdown("""
    - Use specific company names
    - Tool works best with established companies
    - Results may vary based on LinkedIn data availability
    - Review profiles before outreach
    """)
