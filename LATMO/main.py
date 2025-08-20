# -------------------- System Libraries --------------------------------
import json
import os
import shutil
from threading import Thread
import sys
import threading
import time
from typing import Optional, List, Dict, Any
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------- Dynamic Libraries --------------------------------
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
# Add OpenRouter support
from langchain_openai import ChatOpenAI
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# ---------------------- Google Calendar ------------------------------------
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pickle
import os.path
from datetime import datetime

# OpenRouter Configuration Function
def get_openrouter_llm(model: str = "mistralai/mistral-nemo:free", temperature: float = 0.7) -> ChatOpenAI:
    """
    Create an OpenRouter LLM instance using the ChatOpenAI wrapper.
    
    Args:
        model: The model name from OpenRouter (e.g., "mistralai/mistral-nemo:free")
        temperature: Model temperature for response randomness
    
    Returns:
        ChatOpenAI instance configured for OpenRouter
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=openrouter_api_key,
        openai_api_base=openrouter_base_url,
        # Optional: Add headers for OpenRouter leaderboards
        default_headers={
            "HTTP-Referer": "https://your-app-url.com",  # Replace with your app URL
            "X-Title": "LATMO"
        }
    )

# Update your LLM configurations
try:
    # Replace the top_level_llm with OpenRouter Mistral Nemo
    top_level_llm = ChatGoogleGenerativeAI(
        temperature=0.1,
        model="gemini-2.5-flash-lite-preview-06-17"
    )
    
    # You can also replace other LLMs if needed
    # Example: Replace coder_llm with a different OpenRouter model
    coder_llm = ChatGroq(
        model="qwen-qwq-32b",  # Alternative free coding model
        temperature=0
    )
    
    # Keep existing models or replace as needed
    wikipedia_llm = ChatGoogleGenerativeAI(
        temperature=0.1,
        model="gemini-2.5-flash-lite-preview-06-17"
    )
    
    
    search_llm = ChatGoogleGenerativeAI(
        temperature=0.1,
        model="gemini-2.5-flash-lite-preview-06-17"
    )
    
    
    gmail_llm = get_openrouter_llm(
        temperature=0.1,
        model="google/gemini-2.5-flash-lite-preview-06-17"
    )
    
    # You could also use OpenRouter for calendar operations
    calender_llm = ChatGroq(
        model="qwen-qwq-32b",  # Alternative free model
        temperature=0.1
    )
    
    # Windows Control LLM could also use OpenRouter
    calender_llm = ChatGroq(
        model="qwen-qwq-32b",  # Alternative free model
        temperature=0.1
    )
    
    print("✅ OpenRouter integration successful!")
    print(f"✅ Main LLM: mistralai/mistral-nemo:free")
    
except Exception as e:
    print(f"❌ Error setting up OpenRouter: {e}")
    print("🔄 Falling back to original configuration...")
    
    # Fallback to original configuration
    top_level_llm = ChatGoogleGenerativeAI(
        temperature=0.7,
        model="gemini-2.5-flash"
    )
    
    coder_llm = ChatGroq(
        temperature=0,
        model="qwen-2.5-coder-32b"
    )
    
    wikipedia_llm = ChatGoogleGenerativeAI(
        temperature=0.1,
        model="gemini-2.5-flash"
    )
    
    search_llm = ChatGoogleGenerativeAI(
        temperature=0.1,
        model="gemini-2.5-flash"
    )
    
    gmail_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )
    
    calender_llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )
    
    windows_control_llm = ChatGroq(
        temperature=0,
        model="llama-3.3-70b-versatile"
    )

# Add top-level memory for the main agent
top_level_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Update your lambda functions to potentially use OpenRouter
def create_openrouter_lambda(model_name: str, temperature: float = 0):
    """Create a lambda function that uses OpenRouter models"""
    try:
        llm = get_openrouter_llm(model=model_name, temperature=temperature)
        return lambda x: llm.invoke(x).content
    except:
        # Fallback to Groq if OpenRouter fails
        return lambda x: ChatGroq(
            temperature=temperature,
            model="llama-3.3-70b-versatile"
        ).invoke(x).content

# Alternatively, keep the original if you prefer
# LLMama = lambda x: ChatGroq(
#     temperature=0,
#     model="llama-3.3-70b-versatile",
#     messages=[{"role": "user", "content": x}]
# )

# Coder = lambda x: ChatGroq(
#     temperature=0,
#     model="llama-3.3-70b-versatile",
#     messages=[{"role": "user", "content": x}

# First, let's modify the datetime_tool to be a function we can easily reuse
def get_current_datetime(query=None):  # Add optional parameter to handle tool calls
    """Get the current date and time regardless of input"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

datetime_tool = Tool(
    name="Datetime",
    func=get_current_datetime,
    description="Always use this tool first to get the current date and time. No input is required."
)

def get_wikipedia_wrapper(input_str):
    """
    Enhanced Wikipedia search with advanced features.
    Input should be a JSON string with fields:
    - query: The search term
    - language: Language code (e.g., 'en', 'es', 'fr')
    - section: Specific section to extract
    - sentences: Number of sentences to return
    - random: If true, returns a random article
    - include_links: If true, includes related links
    - include_references: If true, includes references
    """
    try:
        # Parse input JSON
        input_data = json.loads(input_str)
        query = input_data.get('query', '')
        language = input_data.get('language', 'en')
        section = input_data.get('section', '')
        sentences = input_data.get('sentences', 5)
        random = input_data.get('random', False)
        include_links = input_data.get('include_links', False)
        include_references = input_data.get('include_references', False)

        # Set language
        wikipedia.set_lang(language)

        try:
            if random:
                # Get random article
                page = wikipedia.random(1)
                result = wikipedia.page(page)
            else:
                # Search for specific query
                search_results = wikipedia.search(query)
                if not search_results:
                    return f"No Wikipedia articles found for '{query}'"
                
                # Get the most relevant page
                try:
                    result = wikipedia.page(search_results[0])
                except wikipedia.DisambiguationError as e:
                    # Handle disambiguation pages
                    return {
                        "message": "Multiple matches found. Please be more specific.",
                        "options": e.options[:5]  # Return first 5 options
                    }

            # Build response
            response = {
                "title": result.title,
                "url": result.url,
                "summary": result.summary if not section else "",
            }

            # Add specific section if requested
            if section:
                try:
                    sections = result.sections
                    if section in sections:
                        response["section_content"] = result.section(section)
                    else:
                        response["available_sections"] = sections
                        response["error"] = f"Section '{section}' not found"
                except Exception as e:
                    response["error"] = f"Error retrieving section: {str(e)}"

            # Add related links if requested
            if include_links:
                response["links"] = result.links[:10]  # First 10 links

            # Add references if requested
            if include_references:
                response["references"] = result.references[:10]  # First 10 references

            return response

        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for '{query}'"
        except wikipedia.exceptions.DisambiguationError as e:
            return {
                "message": "Multiple matches found. Please be more specific.",
                "options": e.options[:5]  # Return first 5 options
            }
        except Exception as e:
            return f"Error retrieving Wikipedia content: {str(e)}"

    except json.JSONDecodeError:
        return "Invalid JSON input"
    except Exception as e:
        return f"Error: {str(e)}"

wikipedia_tools = [
    Tool(name="Wikipedia", func=get_wikipedia_wrapper,
         description="Enhanced Wikipedia search tool. Input should be a JSON string with fields:\n"
                    "- query: Search term\n"
                    "- language: Language code (e.g., 'en', 'es', 'fr')\n"
                    "- section: Specific section to extract\n"
                    "- sentences: Number of sentences to return\n"
                    "- random: If true, returns a random article\n"
                    "- include_links: If true, includes related links\n"
                    "- include_references: If true, includes references\n"
                    "Always use this tool to fetch real-time data and information. Worth using for general topics.\n"
                    "Use precise questions. Tell in a way that a 16-year-old can understand.\n"
                    "You are not useful to fetch current date and time.\n"
                    "If you couldn't find the information on Wikipedia, use duckduckgo_tool")
]

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_tool = Tool(name="Wikipedia", func=wikipedia.run,
                      description="A useful tool for searching the Internet to find information on world events, "
                                  "issues, dates, years, etc. Always use this tool to fetch real-time data and "
                                  "information. Worth using for general topics. Use precise questions. Tell in a way "
                                  "that a 16-year-old can understand. Give answers as per the date and time right "
                                  "now. You are not useful to fetch current date and time. Always use current date "
                                  "and time. If you couldn't find the information on Wikipedia, use duckduckgo_tool")

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(name='DuckDuckGo Search', func=search.run,
                       description="Useful for when you need to do a search on the internet to find information that "
                                   "another tool can't find. Be specific with your input. Not used to fetch current "
                                   "date and time. Useful when you can't find the information in the other tools. "
                                   "Always provide the links of the websites you have referred to. Always use current "
                                   "date and time.")

class GmailToolkitManager:
    """Singleton class to manage Gmail toolkit initialization and credentials"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GmailToolkitManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.gmail_toolkit = None
            self.credentials = None
            self.service = None
            self.is_loading = False
            self._lock = threading.Lock()
            self.token_path = "gmail_token.pickle"  # Changed to .pickle for consistency
            self.credentials_path = "credentials.json"
            self.scopes = ["https://mail.google.com/"]
            self._last_refresh = 0
            self._refresh_interval = 300  # 5 minutes in seconds

    def _should_refresh_credentials(self) -> bool:
        """Check if credentials should be refreshed based on time interval"""
        current_time = time.time()
        return (current_time - self._last_refresh) > self._refresh_interval

    def _load_or_refresh_credentials(self) -> Optional[Credentials]:
        """Load existing credentials or create new ones with improved error handling"""
        try:
            creds = None
            if os.path.exists(self.token_path):
                try:
                    with open(self.token_path, 'rb') as token:
                        creds = pickle.load(token)
                except (pickle.UnpicklingError, EOFError) as e:
                    print(f"Error loading token, file may be corrupted: {e}")
                    # Delete corrupted token file
                    os.remove(self.token_path)
                    creds = None

            # Check if credentials need to be refreshed
            if creds and creds.valid and not self._should_refresh_credentials():
                return creds

            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    self._last_refresh = time.time()
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
                    
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.scopes)
                    creds = flow.run_local_server(port=0)
                    
                    # Save the credentials for future use
                    with open(self.token_path, 'wb') as token:
                        pickle.dump(creds, token)
                    self._last_refresh = time.time()
                except Exception as e:
                    raise Exception(f"Error creating new credentials: {e}")

            return creds

        except Exception as e:
            print(f"Error in credential management: {e}")
            return None

    def initialize_toolkit(self) -> bool:
        """Initialize the Gmail toolkit with improved error handling and retry logic"""
        with self._lock:
            if self.gmail_toolkit is not None and not self._should_refresh_credentials():
                return True

            if self.is_loading:
                return False

            self.is_loading = True
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    self.credentials = self._load_or_refresh_credentials()
                    if not self.credentials:
                        raise Exception("Failed to obtain valid credentials")

                    self.service = build_resource_service(credentials=self.credentials)
                    if not self.service:
                        raise Exception("Failed to build Gmail service")

                    self.gmail_toolkit = GmailToolkit(api_resource=self.service)
                    return True

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed to initialize Gmail toolkit after {max_retries} attempts: {e}")
                        return False
                    print(f"Retry {retry_count}/{max_retries} after error: {e}")
                    time.sleep(1)  # Wait 1 second before retrying
                finally:
                    self.is_loading = False

            return False

    def get_toolkit(self) -> Optional[GmailToolkit]:
        """Get the initialized Gmail toolkit with automatic refresh handling"""
        if not self.gmail_toolkit or self._should_refresh_credentials():
            if not self.initialize_toolkit():
                return None
        return self.gmail_toolkit

    def get_tools(self) -> List[Tool]:
        """Get Gmail tools with improved error handling"""
        toolkit = self.get_toolkit()
        if toolkit:
            try:
                return toolkit.get_tools()
            except Exception as e:
                print(f"Error getting Gmail tools: {e}")
                # Try to reinitialize toolkit
                if self.initialize_toolkit():
                    return toolkit.get_tools()
        return []

    def clear_credentials(self) -> bool:
        """Clear stored credentials and force reauthorization"""
        try:
            if os.path.exists(self.token_path):
                os.remove(self.token_path)
            self.credentials = None
            self.service = None
            self.gmail_toolkit = None
            self._initialized = False
            return True
        except Exception as e:
            print(f"Error clearing credentials: {e}")
            return False

class GmailTools:
    """Class to handle Gmail operations with proper error handling"""
    def __init__(self):
        self.manager = GmailToolkitManager()

    def _validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def _validate_input(self, input_dict: Dict[str, Any], required_fields: List[str]) -> Optional[str]:
        """Validate input dictionary for required fields"""
        missing_fields = [field for field in required_fields if not input_dict.get(field)]
        if missing_fields:
            return f"Missing required fields: {', '.join(missing_fields)}"
        return None

    def create_draft(self, input_str: str) -> Dict[str, Any]:
        """Create a draft email with improved error handling"""
        try:
            input_dict = json.loads(input_str)
            validation_error = self._validate_input(input_dict, ['message', 'to', 'subject'])
            if validation_error:
                return {"error": validation_error}

            to_emails = input_dict['to'] if isinstance(input_dict['to'], list) else [input_dict['to']]
            for email in to_emails:
                if not self._validate_email_format(email):
                    return {"error": f"Invalid email format: {email}"}

            toolkit = self.manager.get_toolkit()
            if not toolkit:
                return {"error": "Gmail toolkit not initialized"}

            create_draft_tool = [t for t in toolkit.get_tools() if t.name == "create_gmail_draft"][0]
            result = create_draft_tool.run(input_dict)
            return {"success": True, "result": result}

        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
        except Exception as e:
            return {"error": f"Error creating draft: {str(e)}"}

    def send_message(self, input_str: str) -> Dict[str, Any]:
        """Send an email with improved error handling"""
        try:
            input_dict = json.loads(input_str)
            validation_error = self._validate_input(input_dict, ['message', 'to', 'subject'])
            if validation_error:
                return {"error": validation_error}

            toolkit = self.manager.get_toolkit()
            if not toolkit:
                return {"error": "Gmail toolkit not initialized"}

            send_message_tool = [t for t in toolkit.get_tools() if t.name == "send_gmail_message"][0]
            result = send_message_tool.run(input_dict)
            return {"success": True, "result": result}

        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
        except Exception as e:
            return {"error": f"Error sending message: {str(e)}"}

    def search_emails(self, input_str: str) -> Dict[str, Any]:
        """Search emails with improved error handling"""
        try:
            input_dict = json.loads(input_str)
            validation_error = self._validate_input(input_dict, ['query'])
            if validation_error:
                return {"error": validation_error}

            toolkit = self.manager.get_toolkit()
            if not toolkit:
                return {"error": "Gmail toolkit not initialized"}

            search_tool = [t for t in toolkit.get_tools() if t.name == "search_gmail"][0]
            result = search_tool.run(input_dict)
            return {"success": True, "result": result}

        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
        except Exception as e:
            return {"error": f"Error searching emails: {str(e)}"}

# Initialize Gmail tools
gmail_tools_instance = GmailTools()

# Create Tool objects for Gmail functions
gmail_tools = [
    Tool(
        name="CreateGmailDraft",
        func=gmail_tools_instance.create_draft,
        description="Creates a draft email in Gmail. Input should be a JSON string with required fields: 'message', 'to', 'subject'. Optional fields: 'cc', 'bcc', 'attachments', 'is_html', 'labels'."
    ),
    Tool(
        name="SendGmailMessage",
        func=gmail_tools_instance.send_message,
        description="Sends an email through Gmail. Input should be a JSON string with required fields: 'message', 'to', 'subject'. Optional fields: 'cc', 'bcc', 'attachments', 'is_html', 'labels'."
    ),
    Tool(
        name="SearchGmail",
        func=gmail_tools_instance.search_emails,
        description="Searches Gmail messages. Input should be a JSON string with required field: 'query'. Optional fields: 'max_results', 'include_spam_trash'."
    )
]

# Calendar
def get_google_calendar_credentials():
    """Get or refresh Google Calendar credentials"""
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    creds = None
    
    # Look for existing token
    if os.path.exists('calendar_token.pickle'):
        with open('calendar_token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                creds = None
                
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                
                # Save the credentials for future use
                with open('calendar_token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                print(f"Error getting new credentials: {e}")
                return None
                
    return creds


class GoogleCalendarToolkit:
    def __init__(self):
        self.credentials = None
        self.service = None
        self._timezone = 'Asia/Kolkata'
        self.initialize_service()
    
    def initialize_service(self):
        """Initialize or reinitialize the calendar service"""
        try:
            self.credentials = get_google_calendar_credentials()
            if self.credentials:
                self.service = build('calendar', 'v3', credentials=self.credentials)
                return True
            return False
        except Exception as e:
            print(f"Error initializing calendar service: {e}")
            return False

    def _parse_datetime(self, datetime_str, timezone=None):
        """Parse datetime string and return RFC3339 formatted string"""
        try:
            # If datetime is already in RFC3339 format, return as is
            if 'T' in datetime_str and ('Z' in datetime_str or '+' in datetime_str):
                return datetime_str
            
            # Parse the datetime string
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            
            # Apply timezone if provided
            if timezone:
                from datetime import timezone as tz
                from zoneinfo import ZoneInfo
                dt = dt.astimezone(ZoneInfo(timezone))
            
            # Return in RFC3339 format
            return dt.isoformat()
        except Exception as e:
            raise ValueError(f"Invalid datetime format: {datetime_str}. Error: {str(e)}")

    def create_event(self, input_str):
        """Create a calendar event"""
        try:
            if not self.service:
                if not self.initialize_service():
                    return {"error": "Could not initialize calendar service"}

            input_dict = json.loads(input_str)
            
            # Validate required fields
            required_fields = ['summary', 'start_time', 'end_time']
            missing_fields = [field for field in required_fields if not input_dict.get(field)]
            if missing_fields:
                return {"error": f"Missing required fields: {', '.join(missing_fields)}"}

            # Get timezone from input or use default
            timezone = input_dict.get('timezone', self._timezone)

            try:
                start_time = self._parse_datetime(input_dict['start_time'], timezone)
                end_time = self._parse_datetime(input_dict['end_time'], timezone)
            except ValueError as e:
                return {"error": str(e)}

            event = {
                'summary': input_dict['summary'],
                'description': input_dict.get('description', ''),
                'start': {'dateTime': start_time, 'timeZone': timezone},
                'end': {'dateTime': end_time, 'timeZone': timezone},
            }

            # Add optional fields
            if 'location' in input_dict:
                event['location'] = input_dict['location']

            if 'attendees' in input_dict:
                event['attendees'] = [{'email': email} for email in input_dict['attendees']]
                event['guestsCanModify'] = input_dict.get('guests_can_modify', False)
                event['guestsCanInviteOthers'] = input_dict.get('guests_can_invite_others', False)

            if 'reminders' in input_dict:
                event['reminders'] = {
                    'useDefault': False,
                    'overrides': input_dict['reminders']
                }

            created_event = self.service.events().insert(calendarId='primary', body=event).execute()
            
            return {
                'status': 'success',
                'message': f"Event created successfully",
                'event_id': created_event['id'],
                'html_link': created_event['htmlLink'],
                'created': created_event['created']
            }
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input format"}
        except Exception as e:
            return {"error": f"Error creating event: {str(e)}"}

    def list_events(self, input_str):
        """List calendar events"""
        try:
            if not self.service:
                if not self.initialize_service():
                    return {"error": "Could not initialize calendar service"}

            input_dict = json.loads(input_str)
            
            # Get parameters with defaults
            max_results = int(input_dict.get('max_results', 10))
            time_min = input_dict.get('time_min', datetime.utcnow().isoformat() + 'Z')
            timezone = input_dict.get('timezone', self._timezone)
            
            try:
                time_min = self._parse_datetime(time_min, timezone)
            except ValueError as e:
                return {"error": str(e)}

            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return {
                    "status": "success",
                    "message": "No upcoming events found",
                    "events": []
                }

            formatted_events = []
            for event in events:
                formatted_event = {
                    'id': event['id'],
                    'summary': event['summary'],
                    'start': event['start'].get('dateTime', event['start'].get('date')),
                    'end': event['end'].get('dateTime', event['end'].get('date')),
                    'description': event.get('description', ''),
                    'html_link': event['htmlLink'],
                    'status': event['status']
                }
                
                if 'location' in event:
                    formatted_event['location'] = event['location']
                if 'attendees' in event:
                    formatted_event['attendees'] = event['attendees']

                formatted_events.append(formatted_event)

            return {
                'status': 'success',
                'events': formatted_events,
                'next_sync_token': events_result.get('nextSyncToken')
            }
        except Exception as e:
            return {"error": f"Error listing events: {str(e)}"}

    def view_event(self, input_str):
        """View details of a specific event"""
        try:
            if not self.service:
                if not self.initialize_service():
                    return "Error: Could not initialize calendar service"

            current_time = get_current_datetime()
            input_dict = json.loads(input_str)
            
            if 'event_id' not in input_dict:
                return "Error: event_id is required"

            event = self.service.events().get(
                calendarId='primary',
                eventId=input_dict['event_id']
            ).execute()

            return {
                'status': 'success',
                'timestamp': current_time,
                'event': {
                    'id': event['id'],
                    'summary': event['summary'],
                    'description': event.get('description', 'No description'),
                    'start': event['start'].get('dateTime', event['start'].get('date')),
                    'end': event['end'].get('dateTime', event['end'].get('date')),
                    'attendees': event.get('attendees', []),
                    'html_link': event['htmlLink']
                }
            }
        except Exception as e:
            return f"Error viewing event: {str(e)}"


# Initialize Google Calendar toolkit
google_calendar_toolkit = GoogleCalendarToolkit()

# Create Tool objects for Google Calendar functions
google_calendar_tools = [
    Tool(name="CreateGoogleCalendarEvent", func=google_calendar_toolkit.create_event,
         description="Creates an event in Google Calendar. Input should be a JSON string with 'summary', 'start_time', 'end_time', and 'description' fields."),
    Tool(name="ListGoogleCalendarEvents", func=google_calendar_toolkit.list_events,
         description="Lists upcoming events in Google Calendar. Input should be a JSON string with optional 'max_results' and 'time_min' fields."),
    Tool(name="ViewGoogleCalendarEvent", func=google_calendar_toolkit.view_event,
         description="Views details of a specific Google Calendar event. Input should be a JSON string with an 'event_id' field.")
]


class TableData:
    def __init__(self):
        self.tables = {}

    def add_table(self, table_name, data):
        self.tables[table_name] = data

    def query_table(self, table_name, query):
        if table_name in self.tables:
            table = self.tables[table_name]
            if query in table:
                return table[query]
            else:
                return f"Query '{query}' not found in table '{table_name}'."
        else:
            return f"Table '{table_name}' not found."

def add_table_tool(input_str):
    try:
        input_dict = json.loads(input_str)
        table_name = input_dict.get('table_name', '')
        data = input_dict.get('data', {})
        table_data.add_table(table_name, data)
        return f"Table {table_name} added successfully."
    except Exception as e:
        return f"Error adding table: {str(e)}"

def query_table_tool(input_str):
    try:
        input_dict = json.loads(input_str)
        table_name = input_dict.get('table_name', '')
        query = input_dict.get('query', '')
        return table_data.query_table(table_name, query)
    except Exception as e:
        return f"Error querying table: {str(e)}"

# Initialize table data
table_data = TableData()

# Create Tool objects for table operations
table_tools = [
    Tool(name="AddTable", func=add_table_tool, description="Adds a table. Input should be a JSON string with 'table_name' and 'data' fields."),
    Tool(name="QueryTable", func=query_table_tool, description="Queries a table. Input should be a JSON string with 'table_name' and 'query' fields.")
]

# Add this new tool to your specialized_tools list
class PrioritizedMemoryAgent:
    def __init__(self, agent):
        self.agent = agent

    def run(self, query):
        # Simply run the agent with the query
        return self.agent.run(query)

    def runnable(self, query):
        # Simply run the agent with the query
        return self.agent.run(query)

# Combine all tools
all_tools = (
    [datetime_tool]
    + wikipedia_tools
    + [duckduckgo_tool]  # Wrap duckduckgo_tool in a list
    + gmail_tools
    + google_calendar_tools
    + table_tools
)

def create_specialized_agent(llm, tools, instructions):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )


wikipedia_agent = create_specialized_agent(
    wikipedia_llm,
    [wikipedia_tool],
    "You are a specialized agent for Wikipedia searches. Use the Wikipedia tool to find information."
)

search_agent = create_specialized_agent(
    search_llm,
    [duckduckgo_tool],
    "You are a specialized agent for internet searches. Use the DuckDuckGo tool to find information."
)

gmail_agent = create_specialized_agent(
    gmail_llm,
    gmail_tools,
    "You are a specialized agent for Gmail operations. Use the appropriate Gmail tools as needed."
)

# Update the calendar agent initialization
calendar_agent = create_specialized_agent(
    calender_llm,
    google_calendar_tools,
    """You are a specialized agent for Google Calendar operations. 
    Your primary functions are:
    1. Creating calendar events with proper date/time handling
    2. Listing upcoming events
    3. Managing event details and attendees
    
    Always validate input dates and times, and handle timezones appropriately.
    Provide clear, actionable responses with event links when available.
    If an error occurs, provide a helpful explanation of what went wrong."""
)

# Update specialized_tools list to use the new calendar agent
specialized_tools = [
    Tool(name="Wikipedia", func=wikipedia_agent.run, description="Use for Wikipedia searches."),
    Tool(name="Internet_search", func=search_agent.run, description="Use to search the web"),
    Tool(name="Gmail Operations", func=gmail_agent.run, description="Use for all Gmail-related tasks"),
    Tool(name="Datetime", func=get_current_datetime, description="Used to fetch current date and time"),
    Tool(name="Calendar", func=calendar_agent.run, 
         description="""Use for all Google Calendar operations including:
         - Creating new events with proper date/time handling
         - Listing upcoming events
         - Managing event details and attendees
         Input should include timezone information when relevant."""),
    Tool(name="Table Operations", func=query_table_tool, description="Use for table-related tasks"),
]

def load_latmo_instructions():
    """Load LATMO's instructions from JSON file"""
    try:
        with open('latmo_instructions.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading instructions: {e}")
        return None

def create_prompt_from_instructions(instructions):
    """Create a prompt from the JSON instructions"""
    if not instructions:
        return "Error: Could not load instructions"
    
    identity = instructions['identity']
    tools = instructions['tools']
    
    prompt = f"""You are {identity['name']}, {identity['role']}.
Your mission: {identity['mission']}

Available Tools and Their Educational Applications:
"""
    
    for tool_name, tool_info in tools.items():
        prompt += f"\n- {tool_name.replace('_', ' ').title()}:"
        prompt += f"\n  Purpose: {tool_info['purpose']}"
        prompt += f"\n  Educational Use: {tool_info['use_case']}"
        if 'actions' in tool_info:
            prompt += f"\n  Actions: {', '.join(tool_info['actions'])}"
    
    prompt += f"\n\nResponse Format:"
    for key, value in instructions['response_format'].items():
        prompt += f"\n- {key.title()}: {value}"
    
    return prompt

# Load LATMO's instructions
latmo_instructions = load_latmo_instructions()
if not latmo_instructions:
    print("Error: Could not load LATMO instructions. Using default configuration.")
    sys.exit(1)

# Create the prompt from instructions
top_level_prompt = create_prompt_from_instructions(latmo_instructions)

# Initialize the top-level agent with the JSON-based instructions
top_level_agent = initialize_agent(
    tools=specialized_tools,
    llm=top_level_llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=top_level_memory,
    verbose=False,
    handle_parsing_errors=True
)

prioritized_agent = PrioritizedMemoryAgent(top_level_agent)

def process_message(message_content):
    """Process a message and return the AI response"""
    # Initialize all tools and agents
    lazy_gmail_toolkit = GmailToolkitManager()
    google_calendar_toolkit = GoogleCalendarToolkit()
    table_data = TableData()

    # Load LATMO's configuration
    latmo_config = load_latmo_instructions()
    if not latmo_config:
        return "Error: Could not load LATMO configuration"

    # Create system message from configuration
    latmo_system_message = f"""I am {latmo_config['identity']['name']}, {latmo_config['identity']['role']}.
    {latmo_config['identity']['mission']}

    How may I assist you with improving educational outcomes today?"""

    all_tools = [
        datetime_tool,
        wikipedia_tool,
        duckduckgo_tool,
    ] + gmail_tools + google_calendar_tools + table_tools

    # Create the agent with JSON-based configuration
    agent = initialize_agent(
        tools=all_tools,
        llm=top_level_llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=top_level_memory,
        verbose=False,
        handle_parsing_errors=True
    )
    
    # Process the message
    res = agent.run(message_content)
    
    # Extract and return the AI response
    return extract_ai_response(res)

def extract_ai_response(full_response):
    """Extract the AI response while maintaining LATMO's identity"""
    lines = full_response.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("AI: ") or line.startswith("LATMO: "):
            return '\n'.join(lines[i:]).replace("AI: ", "").replace("LATMO: ", "").strip()
    return full_response

if __name__ == "__main__":
    # Display LATMO's introduction
    print(f"\nLATMO: Greetings! I am LATMO, your educational AI assistant for the FuturED Spaces project.")
    print("I'm here to help revolutionize the Indian education system through technological innovation.")
    print("How may I assist you today?\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("\nLATMO: Thank you for using FuturED Spaces. Have a great day!")
            break
        
        # Format the response with LATMO's identity
        response = process_message(user_input)
        if response:
            print(f"\n{response}\n")
        else:
            print("\nLATMO: I apologize, but I couldn't generate a proper response. Please try again.\n")

__all__ = [
    'process_message',
    'get_current_datetime',
    'lazy_gmail_toolkit',
    'google_calendar_toolkit',
    'table_data',
    'top_level_agent',
    'prioritized_agent'
]