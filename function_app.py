import azure.functions as func
import logging
from azure.identity import DefaultAzureCredential
import os
import json
import re
from azure.ai.projects import AIProjectClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Regex to match citation markers like 【7:0†source】
_CITATION_PATTERN = re.compile(r"[\u3010][^\u3011]*[\u3011]")

def clean_citation_markers(text: str) -> str:
    """Remove AI-generated citation markers (e.g. 【7:0†source】) from text."""
    return _CITATION_PATTERN.sub("", text).strip()


@app.route(route="workflow_httptrigger")
def workflow_httptrigger(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger function that calls AI Foundry Workflow Agent.
    
    Parameters (query string or JSON body):
        - message: User message/question to send to the workflow agent (required)
        - workflow_name: Name of the workflow to call (optional, uses WORKFLOW_NAME env var as default)
        - threadid: Existing conversation ID for conversation continuity (optional)
        - parameters: JSON object with name-value pairs for message template substitution (optional)
    
    Environment Variables:
        - AIProjectEndpoint: AI Foundry project endpoint URL (required)
        - WORKFLOW_NAME: Default workflow agent name (required if not provided in request)
        - MESSAGE_TEMPLATE: Template string with {variable} placeholders (optional)
          The {message} placeholder is automatically populated from the 'message' parameter.
          Example: "userid: {userid}, username: {username}, question: {message}"
    """
    logging.info('Workflow HTTP trigger function processed a request.')

    # Parse parameters from query string or body
    message = req.params.get('message')
    workflow_name = req.params.get('workflow_name')
    threadid = req.params.get('threadid')
    parameters = None
    
    if not message:
        try:
            req_body = req.get_json()
        except ValueError:
            req_body = None

        if req_body:
            message = req_body.get('message')
            workflow_name = req_body.get('workflow_name')
            threadid = req_body.get('threadid')
            parameters = req_body.get('parameters')  # JSON object with name-value pairs

    if not message:
        return func.HttpResponse(
            json.dumps({
                "error": "Missing required parameter 'message'",
                "usage": "Provide 'message' in query string or request body. Optional: 'workflow_name', 'threadid', 'parameters'"
            }),
            status_code=400,
            mimetype="application/json"
        )

    # Apply message template if configured
    # MESSAGE_TEMPLATE example: "userid: {userid}, username: {username}, question: {message}"
    message_template = os.environ.get("MESSAGE_TEMPLATE")
    if message_template:
        template_vars = {"message": message}
        if parameters and isinstance(parameters, dict):
            template_vars.update(parameters)
        try:
            formatted_message = message_template.format(**template_vars)
            logging.info(f"Applied message template with variables: {list(template_vars.keys())}")
            message = formatted_message
        except KeyError as e:
            return func.HttpResponse(
                json.dumps({
                    "error": f"Missing required parameter for message template: {str(e)}",
                    "provided_parameters": list(template_vars.keys()),
                    "template": message_template
                }),
                status_code=400,
                mimetype="application/json"
            )

    # Get configuration from environment
    endpoint = os.environ.get("AIProjectEndpoint")
    workflow_name = workflow_name or os.environ.get("WORKFLOW_NAME")
    
    if not endpoint:
        logging.error("AIProjectEndpoint must be set in environment variables.")
        return func.HttpResponse(
            json.dumps({"error": "Server configuration error: Missing AIProjectEndpoint"}),
            status_code=500,
            mimetype="application/json"
        )
    
    if not workflow_name:
        logging.error("WORKFLOW_NAME must be set in environment variables or provided in request.")
        return func.HttpResponse(
            json.dumps({"error": "Missing workflow_name parameter or WORKFLOW_NAME environment variable"}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        # Initialize AI Project client
        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )

        # Get the OpenAI client for conversations and responses
        openai_client = project_client.get_openai_client()

        if threadid:
            # Continue existing conversation - verify it exists
            conversation_id = threadid
            openai_client.conversations.retrieve(conversation_id)
        else:
            # Create new conversation for context persistence
            conversation = openai_client.conversations.create()
            conversation_id = conversation.id

        # Call the workflow agent using agent_reference pattern
        # Always pass message as input - workflow agents read from input directly,
        # the conversation ID handles context continuity
        response = openai_client.responses.create(
            conversation=conversation_id,
            input=message,
            extra_body={
                "agent": {"name": workflow_name, "type": "agent_reference"},
            }
        )

        reponse_text = response.output_text

        # Log only Main workflow agent message text
        for item in response.output:
            created_by = getattr(item, 'created_by', None)
            if (created_by and created_by.get('agent', {}).get('name') == workflow_name
                    and item.type == 'message' and item.content):
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        reponse_text = content_item.text  # Capture the last message text from the workflow agent
                        logging.info(f"[{workflow_name}] {content_item.text}")

        # Clean citation markers from the response text
        reponse_text = clean_citation_markers(reponse_text)

        # Build response with exact format requested
        result = {
            "message": reponse_text,
            "threadId": response.conversation.id
        }

        logging.info(f"Workflow response - threadId: {result['threadId']}, message: {result['message']}")

        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            status_code=200,
            mimetype="application/json",
            charset="utf-8"
        )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
