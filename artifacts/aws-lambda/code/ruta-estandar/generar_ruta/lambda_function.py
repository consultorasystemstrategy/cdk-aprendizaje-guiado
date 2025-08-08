import json
import os
import re
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
from aje_libs.bd.helpers.pinecone_helper import PineconeHelper
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
CHATBOT_MODEL_ID = PARAMETER_VALUE["CHATBOT_MODEL_ID"]
CHATBOT_REGION = PARAMETER_VALUE["CHATBOT_REGION"]
CHATBOT_LLM_MAX_TOKENS = int(PARAMETER_VALUE["CHATBOT_LLM_MAX_TOKENS"])
CHATBOT_HISTORY_ELEMENTS = int(PARAMETER_VALUE["CHATBOT_HISTORY_ELEMENTS"])
PINECONE_MAX_RETRIEVE_DOCUMENTS = int(PARAMETER_VALUE["PINECONE_MAX_RETRIEVE_DOCUMENTS"])
PINECONE_MIN_THRESHOLD = float(PARAMETER_VALUE["PINECONE_MIN_THRESHOLD"])
EMBEDDINGS_MODEL_ID = PARAMETER_VALUE["EMBEDDINGS_MODEL_ID"]
EMBEDDINGS_REGION = PARAMETER_VALUE["EMBEDDINGS_REGION"]

# Secrets
#secret_pinecone = SecretsHelper(f"{ENVIRONMENT}/{PROJECT_NAME}/pinecone-api-key2")
secret_pinecone = SecretsHelper(f"{ENVIRONMENT}/agent-resources/pinecone-api-key")
PINECONE_INDEX_NAME = secret_pinecone.get_secret_value("PINECONE_INDEX_NAME")
PINECONE_API_KEY = secret_pinecone.get_secret_value("PINECONE_API_KEY")

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

# Inicialización de recursos
learning_path_table_helper = DynamoDBHelper(
    table_name=DYNAMO_LEARNING_PATH_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="date_time"
)

pinecone_helper = PineconeHelper(
    index_name=PINECONE_INDEX_NAME,
    api_key=PINECONE_API_KEY,
    embeddings_model_id=EMBEDDINGS_MODEL_ID,
    embeddings_region=CHATBOT_REGION,
    max_retrieve_documents=PINECONE_MAX_RETRIEVE_DOCUMENTS,
    min_threshold=PINECONE_MIN_THRESHOLD
)

bedrock_helper = BedrockHelper(region_name=CHATBOT_REGION)

RUTA_PROMPT = """
    ### Instrucción
    Genera una ruta de aprendizaje con {numero_retos} retos centrados en los siguientes temas: {temas_formateados}.

    ### Contexto para uso interno:
    - Competencia: {competencia}
    - Capacidad: {capacidad}
    - Criterio u objetivo de aprendizaje: {criterio}
    - Nivel de complejidad: {complejidad}
    - Temas clave: {temas_formateados}
    - Documentación relevante: {context}

    ### Título de la Ruta de Aprendizaje
    Primero, escribe un **título para la ruta de aprendizaje**, que:
    - Tenga un máximo de **6 palabras**.
    - Use palabras clave relevantes de estos temas: {temas_formateados}.
    - No uses frases genéricas como “Ruta de aprendizaje...”
    - Ejemplo válido: "Domina los Gráficos y Datos"
    @Titulo: [Título breve de la ruta de aprendizaje]

    ### Retos
    @Reto: [Título breve relacionado con un tema distinto]
    @Pregunta: [Pregunta abierta relacionada al tema, nivel {complejidad}]
    @Respuesta Modelo: [Explicación clara, breve y estructurada]
    @Conceptos Claves: [Lista separada por coma de conceptos clave abordados, terminando en punto]

    Proporciona tu respuesta inmediatamente sin ningún preámbulo o información adicional.
"""


SYSTEM_PROMPT2 = """
Eres un asistente llamado {asistente_nombre} que puede ayudar al usuario con sus preguntas usando **únicamente información confiable**.

Contexto del usuario:
- Rol del usuario: {usuario_rol}
- Nombre del usuario: {usuario_nombre}
- Curso: {curso}
- Institución: {institucion}

Instrucciones del modelo:
- Debe proporcionar una respuesta concisa a preguntas sencillas cuando la respuesta se encuentre directamente en los resultados
  de búsqueda. Sin embargo, en el caso de preguntas de sí/no, proporcione algunos detalles.
- Si la pregunta requiere un razonamiento complejo, debe buscar información relevante en los resultados de búsqueda y resumir la
  respuesta basándose en dicha información mediante un razonamiento lógico.
- Si los resultados de búsqueda no contienen información que pueda responder a la pregunta, indique que no pudo encontrar una
  respuesta exacta. Si los resultados de búsqueda son completamente irrelevantes, indique que no pudo encontrar una respuesta exacta y resuma los resultados.
- **NO uses información externa que no esté en los resultados de búsqueda**, excepto para dar explicaciones conceptuales generales del curso **{curso}**.
- **NO inventes información** ni generes contenido fuera del ámbito educativo salvo que el usuario lo solicite explícitamente.
- Mantén **siempre un tono formal, claro y enfocado al ámbito académico**.
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

def upload_ruta(usuario_id: int, silabo_id: int, unidad_id: int, sesion_id: int, prompt_msg: str, ai_msg: str, input_tokens: int, output_tokens: int):
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
            "tipo_metodo_id": 674, # Ruta estándar
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

def get_documents_context(question, data=None):
    """
    Obtiene contexto relevante para una pregunta usando PineconeHelper.
    """
    try:
        logger.info(f"Pregunta: {question}")

        # Si data tiene valor, extraer los resource_id y agregarlos al filtro
        filter_conditions = {}
        if data and 'resources' in data:
            resource_ids = [str(item["resource_id"]) for item in data["resources"]]
            filter_conditions["resource_id"] = {"$in": resource_ids}
            
        logger.info(f"Condiciones de filtro: {filter_conditions}")
        
        # Usar search_by_text en lugar de query_pinecone
        relevant_data = pinecone_helper.search_by_text(
            query_text=question,
            filter_conditions=filter_conditions if filter_conditions else None,
            return_format="text",
            text_field="text"
        )
        
        logger.info(f"Datos relevantes: {relevant_data}\n" + '-'*100)
        return relevant_data
    except Exception as e:
        logger.error(f"Error al obtener el contexto de documentos: {e}")
        return ""
    
def retrieve_context(query_text, resources):
    # Obtener recursos
    if resources:
        if isinstance(resources, str):
            resources = resources.split(",")
        data = {
            "ResourcesIds": [{"resource_id": rid} for rid in resources]
        }             
    else:
        return "No se cuenta con material documental. Genera los retos únicamente con base en tu conocimiento general sobre el tema."

    # Consultar Pinecone
    text_context = get_documents_context(query_text, data)
    return text_context

def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        required_fields = ["UsuarioId", "SilaboId", "UnidadId", "SesionId", "NombreCurso", "Competencia", "Capacidad", "Criterio", "Temas", "Complejidad", "NumeroRetos"]
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
        temas = body.get("Temas", None)
        resources = body.get("ResourcesIds", None)
        complejidad = body["Complejidad"]
        numero_retos = body["NumeroRetos"]

        query_text = (
            criterio + ". Temas: " + ", ".join(temas) + ". En el contexto de: " + capacidad
        )
        logger.info(f"Query_text: {query_text}")

        pinecone_context = retrieve_context(query_text, resources)
        
        # Armar el prompt
        prompt = RUTA_PROMPT.format(
            competencia = competencia,
            capacidad = capacidad,
            criterio = criterio,
            temas_formateados = ', '.join(temas),
            complejidad = complejidad,
            numero_retos = numero_retos,
            context = pinecone_context
        )
        logger.info(f"Ruta prompt: {prompt}")

        response = get_converse_response(prompt=prompt, max_tokens=CHATBOT_LLM_MAX_TOKENS, temperature=0.7)
        learning_path = response['output']['message']['content'][0]['text']
        input_tokens = response['usage']['inputTokens']
        output_tokens = response['usage']['outputTokens']

        upload_ruta(
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