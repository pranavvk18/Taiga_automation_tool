

# config.py
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

@dataclass
class TaigaConfig:
    base_url: str
    username: str
    password: str
    project_slug: str
    
    @classmethod
    def from_env(cls) -> 'TaigaConfig':
        return cls(
            base_url=os.getenv("TAIGA_BASE_URL", "").rstrip('/'),
            username=os.getenv("TAIGA_USERNAME", ""),
            password=os.getenv("TAIGA_PASSWORD", ""),
            project_slug=os.getenv("PROJECT_SLUG", "")
        )
    
    def validate(self) -> None:
        if not all([self.base_url, self.username, self.password, self.project_slug]):
            raise ValueError("Missing required Taiga configuration")

@dataclass
class LLMConfig:
    gemini_api_key: str
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.3
    max_tokens: int = 2048
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
        )
    
    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY")

# models.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class TaskStatus(str, Enum):
    NEW = "New"
    IN_PROGRESS = "In progress"
    READY_FOR_TEST = "Ready for test"
    DONE = "Done"
    ARCHIVED = "Archived"

class UserStory(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10)
    acceptance_criteria: List[str] = Field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    story_points: Optional[int] = Field(None, ge=1, le=21)
    tags: List[str] = Field(default_factory=list)
    
    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError('Description cannot be empty')
        return v.strip()

class TaigaTask(BaseModel):
    subject: str
    description: str
    project_id: int
    status: int = 1  # New
    priority: int = 2  # Normal
    story_points: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    assigned_to: Optional[int] = None

class FeatureRequest(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=20)
    requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    target_users: List[str] = Field(default_factory=list)

# exceptions.py
class TaigaAutomationError(Exception):
    """Base exception for Taiga automation"""
    pass

class TaigaAPIError(TaigaAutomationError):
    """Taiga API related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class LLMError(TaigaAutomationError):
    """LLM related errors"""
    pass

class ValidationError(TaigaAutomationError):
    """Validation related errors"""
    pass

# logger.py
import structlog
import logging
from typing import Any

def setup_logging():
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()

logger = setup_logging()

# taiga_client.py
import requests
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TaigaClient:
    def __init__(self, config: TaigaConfig):
        self.config = config
        self.session = self._create_session()
        self._token: Optional[str] = None
        self._project_id: Optional[int] = None
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def authenticate(self) -> str:
        """Authenticate with Taiga and return auth token"""
        try:
            response = self.session.post(
                f"{self.config.base_url}/api/v1/auth",
                json={
                    "type": "normal",
                    "username": self.config.username,
                    "password": self.config.password
                },
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            self._token = data["auth_token"]
            
            logger.info("Successfully authenticated with Taiga")
            return self._token
            
        except requests.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            raise TaigaAPIError(f"Authentication failed: {e}")
    
    @property
    def token(self) -> str:
        """Get auth token, authenticate if needed"""
        if not self._token:
            self.authenticate()
        return self._token
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_project_id(self) -> int:
        """Get project ID by slug"""
        if self._project_id:
            return self._project_id
            
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = self.session.get(
                f"{self.config.base_url}/api/v1/projects/by_slug",
                params={"slug": self.config.project_slug},
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            self._project_id = data["id"]
            
            logger.info(f"Retrieved project ID: {self._project_id}")
            return self._project_id
            
        except requests.RequestException as e:
            logger.error(f"Failed to get project ID: {e}")
            raise TaigaAPIError(f"Failed to get project ID: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_task(self, task: TaigaTask) -> Dict[str, Any]:
        """Create a task in Taiga"""
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "subject": task.subject,
                "description": task.description,
                "project": self.get_project_id(),
                "status": task.status,
                "priority": task.priority,
                "tags": task.tags
            }
            
            if task.story_points:
                payload["story_points"] = task.story_points
            if task.assigned_to:
                payload["assigned_to"] = task.assigned_to
            
            response = self.session.post(
                f"{self.config.base_url}/api/v1/tasks",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully created task: {task.subject} (ID: {data.get('id')})")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to create task '{task.subject}': {e}")
            raise TaigaAPIError(f"Failed to create task: {e}", 
                              getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
    
    def create_tasks_batch(self, tasks: List[TaigaTask]) -> List[Dict[str, Any]]:
        """Create multiple tasks"""
        results = []
        for task in tasks:
            try:
                result = self.create_task(task)
                results.append(result)
            except TaigaAPIError as e:
                logger.error(f"Failed to create task '{task.subject}': {e}")
                continue
        return results
    
    def get_project_members(self) -> List[Dict[str, Any]]:
        """Get project members for task assignment"""
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = self.session.get(
                f"{self.config.base_url}/api/v1/projects/{self.get_project_id()}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("members", [])
            
        except requests.RequestException as e:
            logger.warning(f"Failed to get project members: {e}")
            return []

# story_generator.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List
import json

class StoryGenerator:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.gemini_api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens
        )
        self.parser = PydanticOutputParser(pydantic_object=UserStory)
        
    def _create_story_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for user story generation"""
        system_template = """You are an expert Product Owner and Business Analyst specializing in Agile methodologies. 
        Your task is to convert feature requests into well-structured user stories following industry best practices.

        Guidelines:
        1. Follow the format: "As a [type of user], I want to [do something] so that [benefit]"
        2. Make stories specific, measurable, achievable, relevant, and time-bound (SMART)
        3. Include clear acceptance criteria
        4. Assign appropriate story points (1, 2, 3, 5, 8, 13, 21)
        5. Add relevant tags for categorization
        6. Consider edge cases and error scenarios
        
        {format_instructions}"""
        
        human_template = """Feature Request: {feature_request}

        Additional Context:
        - Requirements: {requirements}
        - Constraints: {constraints}
        - Target Users: {target_users}
        
        Generate a comprehensive user story for this feature."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_multiple_stories_prompt(self) -> ChatPromptTemplate:
        """Create prompt for generating multiple user stories"""
        system_template = """You are an expert Product Owner who breaks down complex features into manageable user stories.
        
        Guidelines:
        1. Break the feature into 3-8 logical user stories
        2. Ensure stories are independent and deliverable
        3. Order stories by priority and dependencies
        4. Each story should provide value to the user
        5. Include technical tasks if necessary (prefixed with "TECH:")
        
        Return a JSON array of user stories following the specified format.
        
        {format_instructions}"""
        
        human_template = """Feature Request: {feature_request}

        Additional Context:
        - Requirements: {requirements}
        - Constraints: {constraints}
        - Target Users: {target_users}
        
        Break this feature into multiple user stories (3-8 stories)."""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def generate_single_story(self, feature_request: FeatureRequest) -> UserStory:
        """Generate a single user story from feature request"""
        try:
            prompt = self._create_story_prompt()
            
            chain = (
                RunnablePassthrough.assign(
                    format_instructions=lambda _: self.parser.get_format_instructions()
                )
                | prompt
                | self.llm
                | self.parser
            )
            
            result = chain.invoke({
                "feature_request": f"{feature_request.title}\n{feature_request.description}",
                "requirements": "\n".join(feature_request.requirements) if feature_request.requirements else "None specified",
                "constraints": "\n".join(feature_request.constraints) if feature_request.constraints else "None specified",
                "target_users": ", ".join(feature_request.target_users) if feature_request.target_users else "General users"
            })
            
            logger.info(f"Generated user story: {result.title}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate user story: {e}")
            raise LLMError(f"Failed to generate user story: {e}")
    
    def generate_multiple_stories(self, feature_request: FeatureRequest) -> List[UserStory]:
        """Generate multiple user stories from feature request"""
        try:
            # Create custom parser for list of stories
            class UserStoryList(BaseModel):
                stories: List[UserStory]
            
            list_parser = PydanticOutputParser(pydantic_object=UserStoryList)
            prompt = self._create_multiple_stories_prompt()
            
            chain = (
                RunnablePassthrough.assign(
                    format_instructions=lambda _: list_parser.get_format_instructions()
                )
                | prompt
                | self.llm
                | list_parser
            )
            
            result = chain.invoke({
                "feature_request": f"{feature_request.title}\n{feature_request.description}",
                "requirements": "\n".join(feature_request.requirements) if feature_request.requirements else "None specified",
                "constraints": "\n".join(feature_request.constraints) if feature_request.constraints else "None specified",
                "target_users": ", ".join(feature_request.target_users) if feature_request.target_users else "General users"
            })
            
            logger.info(f"Generated {len(result.stories)} user stories")
            return result.stories
            
        except Exception as e:
            logger.error(f"Failed to generate user stories: {e}")
            raise LLMError(f"Failed to generate user stories: {e}")

# task_converter.py
class TaskConverter:
    """Convert user stories to Taiga tasks"""
    
    PRIORITY_MAPPING = {
        TaskPriority.LOW: 1,
        TaskPriority.NORMAL: 2,
        TaskPriority.HIGH: 3,
        TaskPriority.URGENT: 4
    }
    
    @classmethod
    def story_to_task(cls, story: UserStory, project_id: int) -> TaigaTask:
        """Convert UserStory to TaigaTask"""
        
        # Build comprehensive description
        description_parts = [story.description]
        
        if story.acceptance_criteria:
            description_parts.append("\n## Acceptance Criteria:")
            for i, criteria in enumerate(story.acceptance_criteria, 1):
                description_parts.append(f"{i}. {criteria}")
        
        return TaigaTask(
            subject=story.title,
            description="\n".join(description_parts),
            project_id=project_id,
            priority=cls.PRIORITY_MAPPING.get(story.priority, 2),
            story_points=story.story_points,
            tags=story.tags
        )
    
    @classmethod
    def stories_to_tasks(cls, stories: List[UserStory], project_id: int) -> List[TaigaTask]:
        """Convert multiple user stories to Taiga tasks"""
        return [cls.story_to_task(story, project_id) for story in stories]

# automation_engine.py
from typing import Union

class TaigaAutomationEngine:
    """Main automation engine orchestrating the entire process"""
    
    def __init__(self, taiga_config: TaigaConfig, llm_config: LLMConfig):
        self.taiga_client = TaigaClient(taiga_config)
        self.story_generator = StoryGenerator(llm_config)
        self.task_converter = TaskConverter()
        
    def process_feature_request(
        self, 
        feature_request: Union[FeatureRequest, str], 
        generate_multiple: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a feature request end-to-end:
        1. Convert to FeatureRequest if string
        2. Generate user stories
        3. Convert to Taiga tasks
        4. Create tasks in Taiga
        """
        
        # Convert string to FeatureRequest if needed
        if isinstance(feature_request, str):
            feature_request = FeatureRequest(
                title="Auto-generated Feature",
                description=feature_request
            )
        
        logger.info(f"Processing feature request: {feature_request.title}")
        
        try:
            # Generate user stories
            if generate_multiple:
                stories = self.story_generator.generate_multiple_stories(feature_request)
            else:
                story = self.story_generator.generate_single_story(feature_request)
                stories = [story]
            
            logger.info(f"Generated {len(stories)} user stories")
            
            # Convert to Taiga tasks
            project_id = self.taiga_client.get_project_id()
            tasks = self.task_converter.stories_to_tasks(stories, project_id)
            
            # Create tasks in Taiga
            results = self.taiga_client.create_tasks_batch(tasks)
            
            logger.info(f"Successfully created {len(results)} tasks in Taiga")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process feature request: {e}")
            raise
    
    def process_feature_description(self, description: str) -> List[Dict[str, Any]]:
        """Simple interface for processing feature descriptions"""
        return self.process_feature_request(
            FeatureRequest(
                title="Feature Request",
                description=description
            )
        )

# main.py
import sys
from typing import Optional

def create_sample_feature_request() -> FeatureRequest:
    """Create a sample feature request for testing"""
    return FeatureRequest(
        title="User Profile Management System",
        description="""
        Implement a comprehensive user profile management system that allows users to 
        view, edit, and manage their personal information, account settings, and preferences.
        The system should include profile picture upload, password management, and 
        notification preferences.
        """,
        requirements=[
            "Users must be able to upload and crop profile pictures",
            "Password change must require current password verification",
            "Email verification required for email changes",
            "Support for privacy settings (public/private profile)",
            "Activity log showing recent account changes"
        ],
        constraints=[
            "Profile pictures must be under 5MB",
            "Passwords must meet security requirements",
            "Changes must be logged for audit purposes",
            "Must work on mobile and desktop"
        ],
        target_users=[
            "Registered users",
            "Premium subscribers",
            "Admin users"
        ]
    )

def main():
    """Main execution function"""
    try:
        # Load and validate configuration
        taiga_config = TaigaConfig.from_env()
        taiga_config.validate()
        
        llm_config = LLMConfig.from_env()
        llm_config.validate()
        
        logger.info("Starting Taiga automation engine")
        
        # Initialize automation engine
        engine = TaigaAutomationEngine(taiga_config, llm_config)
        
        # Example 1: Process a simple feature description
        simple_feature = "Add dark mode toggle to the application settings page"
        logger.info("Processing simple feature request...")
        results1 = engine.process_feature_description(simple_feature)
        
        # Example 2: Process a complex feature request
        complex_feature = create_sample_feature_request()
        logger.info("Processing complex feature request...")
        results2 = engine.process_feature_request(complex_feature, generate_multiple=True)
        
        # Summary
        total_tasks = len(results1) + len(results2)
        logger.info(f"Automation completed successfully! Created {total_tasks} tasks in Taiga.")
        
        # Print task summaries
        print("\n" + "="*50)
        print("AUTOMATION SUMMARY")
        print("="*50)
        
        print(f"\nSimple Feature ({len(results1)} tasks):")
        for result in results1:
            print(f"  • {result.get('subject', 'Unknown')} (ID: {result.get('id', 'N/A')})")
        
        print(f"\nComplex Feature ({len(results2)} tasks):")
        for result in results2:
            print(f"  • {result.get('subject', 'Unknown')} (ID: {result.get('id', 'N/A')})")
            
    except Exception as e:
        logger.error(f"Automation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# .env.example
"""
# Taiga Configuration
TAIGA_BASE_URL=https://api.taiga.io
TAIGA_USERNAME=your_username
TAIGA_PASSWORD=your_password
PROJECT_SLUG=your-project-slug

# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-pro
GEMINI_TEMPERATURE=0.3
GEMINI_MAX_TOKENS=2048
"""

# cli.py (Optional CLI interface)
import click
from typing import List

@click.group()
def cli():
    """Taiga Automation Tool with LangChain"""
    pass

@cli.command()
@click.argument('description')
@click.option('--multiple', is_flag=True, help='Generate multiple user stories')
@click.option('--dry-run', is_flag=True, help='Generate stories but don\'t create tasks')
def create_from_description(description: str, multiple: bool, dry_run: bool):
    """Create tasks from a feature description"""
    try:
        taiga_config = TaigaConfig.from_env()
        taiga_config.validate()
        
        llm_config = LLMConfig.from_env()
        llm_config.validate()
        
        if dry_run:
            # Only generate stories, don't create tasks
            generator = StoryGenerator(llm_config)
            feature_request = FeatureRequest(title="CLI Feature", description=description)
            
            if multiple:
                stories = generator.generate_multiple_stories(feature_request)
            else:
                stories = [generator.generate_single_story(feature_request)]
            
            click.echo(f"\nGenerated {len(stories)} user stories:")
            for i, story in enumerate(stories, 1):
                click.echo(f"\n{i}. {story.title}")
                click.echo(f"   {story.description}")
                if story.acceptance_criteria:
                    click.echo("   Acceptance Criteria:")
                    for criteria in story.acceptance_criteria:
                        click.echo(f"   - {criteria}")
        else:
            # Full automation
            engine = TaigaAutomationEngine(taiga_config, llm_config)
            results = engine.process_feature_description(description)
            
            click.echo(f"\nSuccessfully created {len(results)} tasks:")
            for result in results:
                click.echo(f"• {result.get('subject')} (ID: {result.get('id')})")
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('feature_file', type=click.File('r'))
def create_from_file(feature_file):
    """Create tasks from a JSON feature request file"""
    try:
        import json
        
        data = json.load(feature_file)
        feature_request = FeatureRequest(**data)
        
        taiga_config = TaigaConfig.from_env()
        taiga_config.validate()
        
        llm_config = LLMConfig.from_env()
        llm_config.validate()
        
        engine = TaigaAutomationEngine(taiga_config, llm_config)
        results = engine.process_feature_request(feature_request)
        
        click.echo(f"\nSuccessfully created {len(results)} tasks:")
        for result in results:
            click.echo(f"• {result.get('subject')} (ID: {result.get('id')})")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main1__':
    cli()