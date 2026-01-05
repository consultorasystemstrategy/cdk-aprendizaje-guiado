import json
import os
import re
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
from .prompts.estrategico_prompt import RUTA_ESTRATEGICO_PROMPT
from .prompts.abp_prompt import RUTA_ABP_PROMPT
from .prompts.clinico_psicologia_prompt import RUTA_CLINICO_PSICOLOGIA_PROMPT
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
DYNAMO_LEARNING_PATH_HISTORY_TABLE = os.environ["DYNAMO_LEARNING_PATH_HISTORY_TABLE"]

# Parameter Store
ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/agent")
PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())
LLM_MODEL_ID = PARAMETER_VALUE["LLM_MODEL_ID"]
LLM_REGION = PARAMETER_VALUE["LLM_REGION"]
LLM_MAX_TOKENS = int(PARAMETER_VALUE["LLM_MAX_TOKENS"])

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

# Inicialización de recursos
learning_path_table_helper = DynamoDBHelper(
    table_name=DYNAMO_LEARNING_PATH_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="date_time"
)

bedrock_helper = BedrockHelper(region_name=LLM_REGION)

def _invoke_prompt(prompt: str, max_tokens: int, temperature: float = 1.0) -> dict:
    """
    Conversa con el modelo de Bedrock usando un prompt.
    
    Parámetros:
    - prompt: texto con las instrucciones del prompt
    - max_tokens: número máximo de tokens de respuesta
    - temperature: control de aleatoriedad
    """

    logger.info(f"Prompt enviado al modelo: {prompt}")
    parameters = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.8
    }

    response = bedrock_helper.converse(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        parameters=parameters
    )
    logger.info(f"Respuesta del modelo: {response}")

    return response

def _upload_ruta(plantilla_id: int, usuario_id: int, silabo_id: int, unidad_id: int, sesion_id: int, prompt_msg: str, ai_msg: str, input_tokens: int, output_tokens: int):
    """
    Sube una ruta a la tabla DynamoDB con los datos especificados.
    """
    try:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TTL en 5 días (432000 segundos)
        # TTL en 7 días (604800 segundos)
        ttl_seconds = 432000
        ttl_timestamp = int((datetime.now() + timedelta(seconds=ttl_seconds)).timestamp())

        item = {
            "tipo_metodo_id": 675, # Método del caso
            "plantilla_id": plantilla_id,
            "usuario_id": usuario_id,
            "date_time": current_datetime,
            "silabo_id": silabo_id,
            "unidad_id": unidad_id,
            "sesion_id": sesion_id,
            "prompt_msg": prompt_msg,
            "ai_msg": ai_msg,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttl": ttl_timestamp
        }

        learning_path_table_helper.put_item(data = item)
        logger.info(f"Elemento subido con éxito: {item}")
    except Exception as e:
        logger.error(f"Error al subir el elemento: {e}")

def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        required_fields = ["UsuarioId", "SilaboId", "UnidadId", "SesionId", "PlantillaId", "NombreCurso", "Competencia", "Capacidad", "Criterio", "Complejidad", "Temas", "Caso"]
        missing_fields = [field for field in required_fields if field not in body]
        if missing_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "success": False,
                    "message": f"Campos requeridos faltantes: {missing_fields}",
                    "error": {
                        "code": "MISSING_FIELDS",
                        "details": f"Campos requeridos faltantes: {missing_fields}"
                    }
                })
            }
        
        user_id = body["UsuarioId"]
        syllabus_event_id = body["SilaboId"]
        unidad_id = body["UnidadId"]
        sesion_id = body["SesionId"]
        plantilla_id = body["PlantillaId"]
        nombre_curso = body["NombreCurso"]
        competencia = body["Competencia"]
        capacidad = body["Capacidad"]
        criterio = body["Criterio"]
        complejidad = body["Complejidad"]
        temas = body.get("Temas", None)
        caso = body["Caso"]

        if plantilla_id == 0: # Estratégico
            prompt = RUTA_ESTRATEGICO_PROMPT.format(
                competencia=competencia,
                capacidad=capacidad,
                criterio=criterio,
                complejidad=complejidad,
                temas_formateados=', '.join(temas),
                caso=caso
            )
        elif plantilla_id == 1: # ABP
            prompt = RUTA_ABP_PROMPT.format(
                competencia=competencia,
                capacidad=capacidad,
                criterio=criterio,
                complejidad=complejidad,
                temas_formateados=', '.join(temas),
                caso=caso
            )
        elif plantilla_id == 2: # Clínico Psicología
            prompt = RUTA_CLINICO_PSICOLOGIA_PROMPT.format(
                competencia=competencia,
                capacidad=capacidad,
                criterio=criterio,
                complejidad=complejidad,
                temas_formateados=', '.join(temas),
                caso=caso
            )

        response = _invoke_prompt(prompt=prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.7)
        learning_path = response['output']['message']['content'][0]['text']
        input_tokens = response['usage']['inputTokens']
        output_tokens = response['usage']['outputTokens']

        _upload_ruta(
            plantilla_id=plantilla_id,
            usuario_id=user_id,
            silabo_id=syllabus_event_id,
            unidad_id=unidad_id,
            sesion_id=sesion_id,
            prompt_msg=prompt,
            ai_msg=learning_path,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "learning_path": learning_path,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        }
    
    except Exception as e:
        logger.error(f"Error en la función Lambda: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "message": str(e)
            })
        }