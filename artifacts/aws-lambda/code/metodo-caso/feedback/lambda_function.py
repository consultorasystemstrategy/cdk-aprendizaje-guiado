import json
import os
import boto3
from aje_libs.common.helpers.bedrock_helper import BedrockHelper
from aje_libs.common.helpers.dynamodb_helper import DynamoDBHelper
from aje_libs.common.helpers.s3_helper import S3Helper
from aje_libs.common.helpers.secrets_helper import SecretsHelper
from aje_libs.common.helpers.ssm_helper import SSMParameterHelper
from aje_libs.common.logger import custom_logger

# Configuración
ENVIRONMENT = os.environ["ENVIRONMENT"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
OWNER = os.environ["OWNER"]

# Parameter Store
ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/agent")
PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())
CHATBOT_MODEL_ID = PARAMETER_VALUE["CHATBOT_MODEL_ID"]
CHATBOT_REGION = PARAMETER_VALUE["CHATBOT_REGION"]
CHATBOT_LLM_MAX_TOKENS = int(PARAMETER_VALUE["CHATBOT_LLM_MAX_TOKENS"])
CHATBOT_HISTORY_ELEMENTS = int(PARAMETER_VALUE["CHATBOT_HISTORY_ELEMENTS"])
PINECONE_MAX_RETRIEVE_DOCUMENTS = int(PARAMETER_VALUE["PINECONE_MAX_RETRIEVE_DOCUMENTS"])
PINECONE_MIN_THRESHOLD = float(PARAMETER_VALUE["PINECONE_MIN_THRESHOLD"])
EMBEDDINGS_MODEL_ID = PARAMETER_VALUE["EMBEDDINGS_MODEL_ID"]
EMBEDDINGS_REGION = PARAMETER_VALUE["EMBEDDINGS_REGION"]

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

# Inicialización de recursos
bedrock_helper = BedrockHelper(region_name=CHATBOT_REGION)

FEEDBACK_PROMPT = """
## Resumen de la tarea:
DEBES redactar una retroalimentación final en un solo párrafo, de forma DIRECTA, para un estudiante que ha respondido a un reto de evaluación, UTILIZANDO exclusivamente las retroalimentaciones previas para su elaboración.

## Información de contexto:
- Curso: {nombre_curso}
- Título del reto: {reto}
- Nivel de complejidad: {complejidad}
- Contexto del caso o situación: {contexto}
- Pregunta que respondió el estudiante: {pregunta}
- Retroalimentaciones previas brindadas según sus respuestas: {feedback}
- Temas clave implicados en la pregunta y que se deben dominar: {temas_formateados}

## Instrucciones para el modelo:
- NO ASUMAS niveles de logro o comprensión que no estén claramente sustentados en la retroalimentación anterior.
- INTEGRA los errores observados como sugerencias de mejora presentadas de manera constructiva.
- INCLUYE recomendaciones específicas orientadas a reforzar los aprendizajes (por ejemplo: repasar conceptos clave, resolver ejercicios adicionales, revisar materiales complementarios).
- EL TONO debe ser profesional, sobrio y orientado al acompañamiento académico.
- NO UTILICES signos de exclamación ni expresiones emotivas o entusiastas.

## Estilo y formato de la respuesta:
- La respuesta DEBE redactarse en un solo párrafo, en texto continuo.
- NO INCLUYAS encabezados, viñetas ni separaciones temáticas.
- MANTÉN un estilo académico, claro y conciso.
"""

def get_converse_response(prompt: str, max_tokens: int, temperature: float = 1.0) -> dict:
    """
    Conversa con el modelo de Bedrock usando un prompt de sistema separado y mensajes estructurados.
    
    Parámetros:
    - prompt: texto con las instrucciones del prompt
    - max_tokens: número máximo de tokens de respuesta
    - temperature: control de aleatoriedad
    """

    logger.info(json.dumps(prompt, indent=2))

    parameters = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.2
    }

    response = bedrock_helper.converse(
        model=CHATBOT_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        parameters=parameters
    )

    return response


def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        required_fields = ["NombreCurso", "Complejidad", "Reto", "Pregunta", "RespuestaModelo", "Temas", "Feedback"]
        missing_fields = [field for field in required_fields if field not in body]
        if missing_fields:
            return {
                "success": False,
                "message": f"Campos requeridos faltantes: {missing_fields}",
                "statusCode": 400,
                "error": {
                    "code": "MISSING_FIELDS",
                    "details": f"Campos requeridos faltantes: {missing_fields}"
                }
            }
        
        nombre_curso = body["NombreCurso"]
        complejidad = body["Complejidad"]
        reto = body["Reto"]
        contexto = body["Contexto"]
        pregunta = body["Pregunta"]
        respuesta_modelo = body["RespuestaModelo"]
        temas = body.get("Temas", None)
        feedback = body.get("Feedback", None)
        
        prompt = FEEDBACK_PROMPT.format(
            nombre_curso=nombre_curso,
            reto=reto,
            complejidad=complejidad,
            contexto=contexto,
            pregunta=pregunta,
            feedback=', '.join(feedback),
            temas_formateados=', '.join(temas)
        )
        response = get_converse_response(prompt = prompt, max_tokens=CHATBOT_LLM_MAX_TOKENS, temperature=0.5)
        feedback_response = response['output']['message']['content'][0]['text']
        input_tokens = response['usage']['inputTokens']
        output_tokens = response['usage']['outputTokens']

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "feedback": feedback_response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        }
        
    except Exception as e:
        logger.error(f"Error en get_history: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "message": str(e)
            })
        }