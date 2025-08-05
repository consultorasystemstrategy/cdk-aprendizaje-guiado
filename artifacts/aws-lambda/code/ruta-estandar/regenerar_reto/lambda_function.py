import json
import os
import boto3
import re
from boto3.dynamodb.conditions import Attr
from datetime import datetime, timedelta
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
DYNAMO_REGENERATED_HISTORY_TABLE = os.environ["DYNAMO_REGENERATED_CHALLENGES_HISTORY_TABLE"]

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

# Inicializar DynamoDBHelper
regenerated_table_helper = DynamoDBHelper(
    table_name=DYNAMO_REGENERATED_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="date_time"
)

bedrock_helper = BedrockHelper(region_name=CHATBOT_REGION)

REGENERAR_RETO_PROMPT_SIN_INDICACIONES = '''
    Eres un experto en pedagogía y en el curso {nombre_curso}. Tu tarea es generar un reto de aprendizaje siguiendo exactamente este formato:

    @Reto: [Título breve del reto relacionado con uno de los temas clave proporcionados]
    @Pregunta: [Pregunta abierta relacionada al tema del reto]
    @Respuesta Modelo: [Explicación clara, breve y estructurada que responda la pregunta]
    @Conceptos Claves: [Lista separada por coma de los conceptos clave abordados en ese reto, y que termine en punto]

    Debes utilizar la siguiente información como base:
    - Competencia: {competencia}
    - Capacidad: {capacidad}
    - Criterio u objetivo de aprendizaje: {criterio}
    - Temas clave: {temas_formateados}

    Pregunta ya utilizada anteriormente (no la repitas):
    "{pregunta}"

    Asegúrate de generar un reto centrado en uno o varios de los temas clave proporcionados. Utiliza un lenguaje claro, técnico y directo. ¡Responde con la mayor precisión posible!
'''

REGENERAR_RETO_PROMPT_BY_INDICACIONES = '''
    Eres un experto en pedagogía y en el curso {nombre_curso}. Tu tarea es generar un reto de aprendizaje siguiendo exactamente este formato:

    @Reto: [Título breve del reto relacionado con uno de los temas clave proporcionados]
    @Pregunta: [Pregunta abierta relacionada al tema del reto]
    @Respuesta Modelo: [Explicación clara, breve y estructurada que responda la pregunta]
    @Conceptos Claves: [Lista separada por coma de los conceptos clave abordados en ese reto, y que termine en punto]

    Debes utilizar la siguiente información como base:
    - Competencia: {competencia}
    - Capacidad: {capacidad}
    - Criterio u objetivo de aprendizaje: {criterio}
    - Temas clave: {temas_formateados}

    Campos proporcionados:
    - Título del reto: "{titulo_reto}" 
    - Pregunta: "{pregunta}"
    - Respuesta modelo: "{respuesta_modelo}"

    {indicaciones}

    Asegúrate de generar un reto centrado en uno o varios de los temas clave proporcionados. Utiliza un lenguaje claro, técnico y directo. ¡Responde con la mayor precisión posible!
'''

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

def upload_reto(usuario_id: int, silabo_id: int, unidad_id: int, sesion_id: int, indicaciones: str, prompt_msg: str, ai_msg: str, input_tokens: int, output_tokens: int):
    """
    Sube un reto a la tabla DynamoDB con los datos especificados.
    """
    try:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TTL en 5 días (432000 segundos)
        # TTL en 7 días (604800 segundos)
        ttl_seconds = 432000
        ttl_timestamp = int((datetime.now() + timedelta(seconds=ttl_seconds)).timestamp())

        item = {
            "usuario_id": usuario_id,
            "date_time": current_datetime,
            "silabo_id": silabo_id,
            "unidad_id": unidad_id,
            "sesion_id": sesion_id,
            "indicaciones": indicaciones,
            "prompt_msg": prompt_msg,
            "ai_msg": ai_msg,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttl": ttl_timestamp
        }

        regenerated_table_helper.put_item(data = item)
        logger.info(f"Elemento subido con éxito: {item}")
    except Exception as e:
        logger.error(f"Error al subir el elemento: {e}")

def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        required_fields = ["UsuarioId", "SilaboId", "UnidadId", "SesionId", "NombreCurso", "Competencia", "Capacidad", "Criterio", "TituloReto", "Pregunta", "RespuestaModelo", "Temas", "Indicaciones"]
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
        
        user_id = body["UsuarioId"]
        syllabus_event_id = body["SilaboId"]
        unidad_id = body["UnidadId"]
        sesion_id = body["SesionId"]
        nombre_curso = body["NombreCurso"]
        competencia = body["Competencia"]
        capacidad = body["Capacidad"]
        criterio = body["Criterio"]
        titulo_reto = body["TituloReto"]
        pregunta = body["Pregunta"]
        respuesta_modelo = body["RespuestaModelo"]
        temas = body.get("Temas", None)
        indicaciones = body["Indicaciones"]

        prompt = REGENERAR_RETO_PROMPT_BY_INDICACIONES.format(
            nombre_curso = nombre_curso,
            competencia = competencia,
            capacidad = capacidad,
            criterio = criterio,
            titulo_reto = titulo_reto,
            pregunta = pregunta,
            respuesta_modelo = respuesta_modelo,
            temas_formateados = ', '.join(temas),
            indicaciones = indicaciones
        )

        response = get_converse_response(prompt = prompt, max_tokens=CHATBOT_LLM_MAX_TOKENS, temperature=0.7)
        regenerated_challenge = response['output']['message']['content'][0]['text']
        input_tokens = response['usage']['inputTokens']
        output_tokens = response['usage']['outputTokens']

        # Guardar en historial
        upload_reto(
            usuario_id=user_id,
            silabo_id=syllabus_event_id,
            unidad_id=unidad_id,
            sesion_id=sesion_id,
            indicaciones=indicaciones,
            prompt_msg=prompt,
            ai_msg=regenerated_challenge,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "regenerated_challenge": regenerated_challenge,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "message": str(e)
            })
        }