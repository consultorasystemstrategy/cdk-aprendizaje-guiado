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
DYNAMO_EVALUATION_HISTORY_TABLE = os.environ["DYNAMO_EVALUATION_HISTORY_TABLE"]

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
evaluation_table_helper = DynamoDBHelper(
    table_name=DYNAMO_EVALUATION_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="date_time"
)

bedrock_helper = BedrockHelper(region_name=CHATBOT_REGION)

SCORE_PROMPT = """
    Eres un experto evaluador académico en {nombre_curso}. Tu tarea es asignar un puntaje objetivo entre 0.0 y 1.0 a la respuesta de un estudiante, comparándola con una respuesta modelo, según los siguientes criterios académicos. Debes tener en cuenta también el **contexto** en el que se formula la pregunta.

    Contexto:
    {contexto}

    Pregunta:
    {pregunta}

    Respuesta del estudiante:
    {respuesta_usuario}

    Respuesta modelo esperada:
    {respuesta_modelo}

    Temas clave esperados:
    {temas_formateados}

    Criterios de evaluación:
    1. Precisión conceptual.
    2. Cobertura de los puntos clave.
    3. Claridad y coherencia.
    4. Equivalencia semántica.
    5. Relevancia respecto a los temas clave y el contexto proporcionado.

    Formato de salida:
    - Devuelve **solo** un número decimal entre 0.0 y 1.0 con dos decimales.
    - **No agregues explicaciones, etiquetas, comentarios ni palabras adicionales.**
"""

FEEDBACK_ALL_PROMPT = """
    Eres un docente experto en retroalimentación pedagógica. Tu tarea es ayudar a un estudiante que no respondió correctamente una pregunta de evaluación.

    Contexto de retroalimentación:
    - Curso: {nombre_curso}
    - Complejidad: {complejidad}
    - Contexto del caso o situación: {contexto}
    - Pregunta original: {pregunta}
    - Respuesta modelo esperada: {respuesta_modelo}
    - Respuesta del estudiante: {respuesta_usuario}
    - Temas clave involucrados: {temas_formateados}

    El puntaje de su respuesta fue bajo.

    Tu tarea es:

    1. Generar una retroalimentación breve (máximo 3 líneas) que oriente al estudiante sobre su error, omisión o confusión.
    - Si su respuesta es parcialmente correcta, enfócate en aclarar lo que faltó o se interpretó mal.
    - Si su respuesta no evidencia comprensión, enfócate en explicar lo esencial del concepto o propósito del tema evaluado.

    2. Reformular una nueva pregunta (no repetir la misma), tomando en cuenta el **contexto proporcionado**:
    - Si el error fue parcial, mantener la complejidad, pero cambiar el enfoque o el caso para reforzar el concepto omitido.
    - Si el estudiante no demuestra comprensión alguna, reducir ligeramente la complejidad y asegurar que se toquen los conceptos clave omitidos.

    La nueva pregunta debe ser diferente de la original y enfocarse en ayudar al estudiante a superar su dificultad.

    3. Brindar una respuesta modelo clara, precisa y alineada con la nueva pregunta.

    4. Actualizar la lista de conceptos clave en función de la nueva pregunta formulada.

    Devuelve el resultado con el siguiente formato (sin agregar explicaciones adicionales):

    @Feedback: [Escribe aquí la retroalimentación]
    @Pregunta: [Escribe aquí la nueva pregunta reformulada]
    @Respuesta Modelo: [Escribe aquí la nueva respuesta modelo]
    @Conceptos Claves: [Lista separada por comas, terminando en punto]
"""

FEEDBACK_PROMPT = """
    Eres un docente experto en retroalimentación pedagógica. Tu tarea es generar una retroalimentación breve y profesional.

    Contexto de retroalimentación:
    - Curso: {nombre_curso}
    - Complejidad: {complejidad}
    - Contexto del caso o situación: {contexto}
    - Pregunta original: {pregunta}
    - Respuesta modelo esperada: {respuesta_modelo}
    - Respuesta del estudiante: {respuesta_usuario}
    - Temas clave involucrados: {temas_formateados}

    El puntaje de su respuesta fue alto. 

    Tu tarea es:

    1. Generar una retroalimentación breve (máximo 3 líneas) que:
    - Reconozca de forma directa qué concepto o proceso ha comprendido correctamente el estudiante **en función del contexto proporcionado**.
    - Incluya una sugerencia concreta y moderada para seguir profundizando o aplicando lo aprendido dentro de situaciones similares o más complejas.

    Devuelve el resultado con el siguiente formato (sin agregar explicaciones adicionales):

    @Feedback: [Escribe aquí la retroalimentación]
"""

SCORE_PROMPT2 = """
## Resumen de la tarea:
Eres un evaluador académico experto en el curso {nombre_curso}. Tu tarea es asignar un puntaje objetivo entre 0.0 y 1.0 a la respuesta de un estudiante, comparándola con una respuesta modelo, según criterios académicos establecidos.

## Información de contexto:
- A continuación, se presenta el contexto, la pregunta, la respuesta del estudiante y la respuesta modelo esperada.
- También se indican los temas clave que deben estar presentes en la respuesta.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta del estudiante:
{respuesta_usuario}

Respuesta modelo esperada:
{respuesta_modelo}

Temas clave esperados:
{temas_formateados}

## Instrucciones para el modelo:
- Evalúa la respuesta del estudiante considerando los siguientes criterios:
  1. Precisión conceptual.
  2. Cobertura de los puntos clave.
  3. Claridad y coherencia.
  4. Equivalencia semántica con la respuesta modelo.
  5. Relevancia con respecto a los temas clave.
- DEBES penalizar respuestas que estén vacías, contengan solo signos, emojis o contenido irrelevante.
- NO DEBES otorgar puntajes altos a respuestas que carezcan de contenido académico significativo.
- Usa todo el rango de la escala de 0.0 a 1.0 de forma adecuada.

## Requisitos de estilo y formato de la respuesta:
- DEBES responder SOLO con un número decimal entre 0.0 y 1.0, con dos decimales (ejemplo: 0.75).
- NO DEBES incluir explicaciones, etiquetas, comentarios ni ningún otro texto adicional.
- NO USES markdown ni comillas. NO uses bloques ``` de ningún tipo.
- Responde ÚNICAMENTE el número, con dos decimales.
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

def upload_evaluar(reto_ejecucion_id: str, usuario_id: int, silabo_id: int, unidad_id: int, sesion_id: int, score: str, prompt_msg: str, ai_msg: str, input_tokens: int, output_tokens: int):
    """
    Sube una evaluación realizada a la tabla DynamoDB con los datos especificados.
    """
    try:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TTL en 5 días (432000 segundos)
        # TTL en 7 días (604800 segundos)
        ttl_seconds = 432000
        ttl_timestamp = int((datetime.now() + timedelta(seconds=ttl_seconds)).timestamp())

        item = {
            "tipo_metodo_id": 674,
            "reto_ejecucion_id": reto_ejecucion_id,
            "usuario_id": usuario_id,
            "date_time": current_datetime,
            "silabo_id": silabo_id,
            "unidad_id": unidad_id,
            "sesion_id": sesion_id,
            "score": score,
            "prompt_msg": prompt_msg,
            "ai_msg": ai_msg,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttl": ttl_timestamp
        }

        evaluation_table_helper.put_item(data = item)
        logger.info(f"Elemento subido con éxito: {item}")
    except Exception as e:
        logger.error(f"Error al subir el elemento: {e}")

def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        required_fields = ["RetoEjecucionId", "UsuarioId", "SilaboId", "UnidadId", "SesionId", "NombreCurso", "Complejidad", "Contexto", "Pregunta", "RespuestaModelo", "RespuestaUsuario", "Temas", "Umbral"]
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
        
        reto_ejecucion_id = body["RetoEjecucionId"]
        user_id = body["UsuarioId"]
        syllabus_event_id = body["SilaboId"]
        unidad_id = body["UnidadId"]
        sesion_id = body["SesionId"]
        nombre_curso = body["NombreCurso"]
        complejidad = body["Complejidad"]
        contexto = body["Contexto"]
        pregunta = body["Pregunta"]
        respuesta_modelo = body["RespuestaModelo"]
        respuesta_usuario = body["RespuestaUsuario"]
        temas = body.get("Temas", None)
        umbral = body["Umbral"]
        
        prompt = SCORE_PROMPT2.format(
            nombre_curso = nombre_curso,
            contexto = contexto,
            pregunta = pregunta,
            respuesta_usuario = respuesta_usuario,
            respuesta_modelo = respuesta_modelo,
            temas_formateados = ', '.join(temas)
        )
        
        response = get_converse_response(prompt=prompt, max_tokens=5, temperature=0.0)
        logger.info(f"Response Score from Bedrock: {response}")
        score_response = response['output']['message']['content'][0]['text']

        # Intentar detectar el número sin etiqueta
        primera_linea = score_response.strip().splitlines()[0]
        match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*$", primera_linea)

        if match:
            score = float(match.group(1))
                
            # Comparar con umbral
            if score < umbral:
                prompt = FEEDBACK_ALL_PROMPT.format(
                    nombre_curso = nombre_curso,
                    complejidad = complejidad,
                    contexto = contexto,
                    pregunta = pregunta,
                    respuesta_modelo = respuesta_modelo,
                    respuesta_usuario = respuesta_usuario,
                    temas_formateados = ', '.join(temas)
                )

            else:
                prompt = FEEDBACK_PROMPT.format(
                    nombre_curso = nombre_curso,
                    complejidad = complejidad,
                    contexto = contexto,
                    pregunta = pregunta,
                    respuesta_modelo = respuesta_modelo,
                    respuesta_usuario = respuesta_usuario,
                    temas_formateados = ', '.join(temas)
                )

            response = get_converse_response(prompt = prompt, max_tokens=CHATBOT_LLM_MAX_TOKENS, temperature = 0.7)
            feedback = response['output']['message']['content'][0]['text']
            input_tokens = response['usage']['inputTokens']
            output_tokens = response['usage']['outputTokens']
        
            upload_evaluar(
                reto_ejecucion_id=reto_ejecucion_id,
                usuario_id=user_id,
                silabo_id=syllabus_event_id,
                unidad_id=unidad_id,
                sesion_id=sesion_id,
                score=str(score),
                prompt_msg=prompt,
                ai_msg=feedback,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        else:
            score = 0
            feedback = "No se encontró el puntaje."
            input_tokens = 0
            output_tokens = 0

            upload_evaluar(
                reto_ejecucion_id=reto_ejecucion_id,
                usuario_id=user_id,
                silabo_id=syllabus_event_id,
                unidad_id=unidad_id,
                sesion_id=sesion_id,
                score=str(score),
                prompt_msg=prompt, # Enviará el prompt utilizado para obtener el score
                ai_msg=feedback,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "score": score,
                "feedback": feedback,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        }
        
    except Exception as e:
        logger.error(f"Error en delete_history: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "message": str(e)
            })
        }