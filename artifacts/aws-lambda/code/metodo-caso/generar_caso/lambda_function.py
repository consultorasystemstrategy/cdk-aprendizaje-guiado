import json
import os
import re
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
from .prompts.estrategico_prompt import CASO_ESTRATEGICO_PROMPT
from .prompts.abp_prompt import ABP_PROMPT
from .prompts.clinico_psicologia_prompt import CLINICO_PSICOLOGIA_PROMPT
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
DYNAMO_CASE_HISTORY_TABLE = os.environ["DYNAMO_CASE_HISTORY_TABLE"]

# Parameter Store
ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/agent")
PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())
CHATBOT_MODEL_ID = PARAMETER_VALUE["CHATBOT_MODEL_ID"]
CHATBOT_REGION = PARAMETER_VALUE["CHATBOT_REGION"]
CHATBOT_LLM_MAX_TOKENS = int(PARAMETER_VALUE["CHATBOT_LLM_MAX_TOKENS"])

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

# Inicialización de recursos
case_history_table_helper = DynamoDBHelper(
    table_name=DYNAMO_CASE_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="date_time"
)

bedrock_helper = BedrockHelper(region_name=CHATBOT_REGION)

'''
CASO_ESCOLAR_PROMPT = """
    ## Tarea
    Escribe un caso breve para estudiantes de nivel primaria. El caso debe ser claro, cercano y sin soluciones ni juicios.

    ## Instrucciones
    1. Si el usuario ha proporcionado un texto base, utilízalo para identificar los siguientes elementos:
        - Nombre de la institución o empresa ficticia y su entorno (escuela, comunidad, emprendimiento)
        - Dilema o problema principal que pueda comprender un estudiante de primaria
        - Propósito o misión que guíe a los personajes
        - Otros datos relevantes para construir el caso

    2. Si no se ha proporcionado ningún texto base, **genera un caso original** relacionado con los datos curriculares proporcionados (curso, competencia, capacidad, criterio, nivel de complejidad y temas clave).

    3. Ajusta el nivel de detalle, el lenguaje y el dilema según el nivel de complejidad especificado ({complejidad}), manteniendo una narrativa comprensible y motivadora para estudiantes de primaria.

    4. El dilema debe girar en torno a los temas clave ({temas_formateados}) e implicar la competencia, capacidad y criterio proporcionados, sin nombrarlos explícitamente.

    5. El caso debe cubrir los siguientes aspectos mediante el análisis:
        - Identificación del problema
        - Análisis de causas
        - Alternativas posibles
        - Diseño de una solución o plan de acción

    6. Incluye los siguientes elementos:
        - 2-3 personajes (niños, docentes, familiares u otros) con perspectivas distintas
        - Datos simples y concretos (números, ejemplos, frases de los personajes, emociones, situaciones)
        - 2-3 alternativas relacionadas con los temas clave, explicando sus ventajas, desventajas y posibles consecuencias
        - Información mínima para actuar: qué hacen, dónde están, qué recursos tienen, qué tiempos manejan

    7. No incluir nota didáctica ni cierre instructivo.

    ## Datos curriculares
    - Curso: {nombre_curso}
    - Competencia: {competencia}
    - Capacidad: {capacidad}
    - Criterio: {criterio}
    - Complejidad: {complejidad}
    - Temas clave: {temas_formateados}

    ## Texto base del usuario (puede estar vacío)
    {contexto}

    ## Estructura esperada del caso
    1. Título
    2. Resumen
    3. Contexto
    4. Datos clave
    5. Problema central
    6. Personajes
    7. Alternativas
    8. Información operativa mínima
"""

CASO_AVANZADO_PROMPT = """
    ## Tarea
    Escribe un caso estratégico breve, al estilo Harvard/IESE, para análisis individual. El caso debe ser claro, profesional y sin soluciones ni juicios.

    ## Instrucciones
    1. Si el usuario ha proporcionado un texto base, utilízalo para identificar los siguientes elementos:
        - Nombre de la empresa y sector
        - Dilema o problema estratégico
        - Misión o propósito institucional
        - Otros datos relevantes para la construcción del caso

    2. Si no se ha proporcionado ningún texto base, **genera un caso estratégico original** relacionado con los datos curriculares (curso, competencia, capacidad, criterio, nivel de complejidad y temas clave).

    3. Ajusta el nivel de detalle, la complejidad del dilema y la profundidad del contexto según el nivel de complejidad especificado ({complejidad}).

    4. El dilema debe girar en torno a los temas clave ({temas_formateados}) e implicar la competencia, capacidad y criterio proporcionados, sin nombrarlos explícitamente.

    5. El caso debe cubrir los siguientes aspectos mediante el análisis:
        - Identificación del problema
        - Análisis causal
        - Alternativas estratégicas
        - Diseño del plan de acción

    6. Incluye los siguientes elementos:
        - 2-3 actores con perspectivas distintas
        - Datos cuantitativos y cualitativos (cifras, indicadores, conflictos, citas)
        - 2-3 alternativas estratégicas vinculadas a los temas clave, incluyendo ventajas, limitaciones, áreas y efectos
        - Información operativa mínima: cargos, áreas, recursos y plazos

    7. No incluir nota didáctica ni cierre instructivo.

    ## Datos curriculares
    - Curso: {nombre_curso}
    - Competencia: {competencia}
    - Capacidad: {capacidad}
    - Criterio: {criterio}
    - Complejidad: {complejidad}
    - Temas clave: {temas_formateados}

    ## Texto base del usuario (puede estar vacío)
    {contexto}

    ## Estructura esperada del caso
    1. Título
    2. Resumen
    3. Contexto
    4. Datos clave
    5. Problema central
    6. Actores
    7. Alternativas estratégicas
    8. Información operativa mínima
"""
'''

def invoke_prompt(prompt: str, max_tokens: int, temperature: float = 1.0) -> dict:
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
        model=CHATBOT_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        parameters=parameters
    )
    logger.info(f"Respuesta del modelo: {response}")

    return response

def upload_caso(plantilla_id: int, usuario_id: int, silabo_id: int, unidad_id: int, sesion_id: int, prompt_msg: str, ai_msg: str, input_tokens: int, output_tokens: int):
    """
    Sube un caso a la tabla DynamoDB con los datos especificados.
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

        case_history_table_helper.put_item(data = item)
        logger.info(f"Elemento subido con éxito: {item}")
    except Exception as e:
        logger.error(f"Error al subir el elemento: {e}")

def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        required_fields = ["UsuarioId", "SilaboId", "UnidadId", "SesionId", "PlantillaId", "Contexto", "NombreCurso", "Competencia", "Capacidad", "Criterio", "Complejidad", "Temas"]
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
        contexto = body["Contexto"]
        nombre_curso = body["NombreCurso"]
        competencia = body["Competencia"]
        capacidad = body["Capacidad"]
        criterio = body["Criterio"]
        complejidad = body["Complejidad"]
        temas = body.get("Temas", None)

        if plantilla_id == 0: # Estratégico
            prompt = CASO_ESTRATEGICO_PROMPT.format(
                contexto=contexto,
                nombre_curso=nombre_curso,
                competencia=competencia,
                capacidad=capacidad,
                criterio=criterio,
                complejidad=complejidad,
                temas_formateados=', '.join(temas),
            )
        elif plantilla_id == 1: # ABP
            prompt = ABP_PROMPT.format(
                contexto=contexto,
                nombre_curso=nombre_curso,
                competencia=competencia,
                capacidad=capacidad,
                criterio=criterio,
                complejidad=complejidad,
                temas_formateados=', '.join(temas),
            )
        elif plantilla_id == 2: # Clínico Psicología
            prompt = CLINICO_PSICOLOGIA_PROMPT.format(
                contexto=contexto,
                nombre_curso=nombre_curso,
                competencia=competencia,
                capacidad=capacidad,
                criterio=criterio,
                complejidad=complejidad,
                temas_formateados=', '.join(temas),
            )

        response = invoke_prompt(prompt=prompt, max_tokens=CHATBOT_LLM_MAX_TOKENS, temperature=0.7)
        case = response['output']['message']['content'][0]['text']
        input_tokens = response['usage']['inputTokens']
        output_tokens = response['usage']['outputTokens']

        # Guardar en historial
        upload_caso(
            plantilla_id=plantilla_id,
            usuario_id=user_id,
            silabo_id=syllabus_event_id,
            unidad_id=unidad_id,
            sesion_id=sesion_id,
            prompt_msg=prompt,
            ai_msg=case,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "case": case,
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