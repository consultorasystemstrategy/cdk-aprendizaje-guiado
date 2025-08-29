import json
import os
import re
from boto3.dynamodb.conditions import Key, Attr
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

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

# Inicialización de recursos
learning_path_table_helper = DynamoDBHelper(
    table_name=DYNAMO_LEARNING_PATH_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="date_time"
)

bedrock_helper = BedrockHelper(region_name=CHATBOT_REGION)

RUTA_PROMPT = """
    ## Tarea
    Generar cinco retos formativos alineados con las etapas del análisis de casos individuales, utilizando el caso proporcionado y los datos curriculares. Cada reto debe evaluar una habilidad específica por etapa, usando el caso como base y respetando la estructura detallada.

    ## Formato obligatorio
    Utilice exactamente el siguiente formato:

    @Nombre: [Título general de la ruta, máximo 6 palabras. Incluya al menos una palabra clave de los temas clave.]

    (Después, por cada reto:)

    @Reto: [Título breve del desafío]  
    @Contexto: [Máx. 60 palabras. Incluya actores, hechos o tensiones clave del caso. En el reto 5 debe incluir decisiones tomadas, actores clave y plazos.]  
    @Pregunta: [Una sola línea con la pregunta principal contextualizada + subpregunta que aplique directamente uno de los temas clave listados en {temas_formateados}.]  
    @Respuesta Modelo: [Respuesta clara, analítica y contextual.]  
    @Conceptos Clave: [**Inicie con el mismo tema clave exacto usado en la subpregunta.** Luego, agregue otros conceptos o herramientas complementarias. Separe por comas y termine en punto.]

    ## Ejemplo

    @Nombre: Explorando patrones en lectura escolar  

    @Reto: Diagnóstico del caos en datos  
    @Contexto: La bibliotecaria enfrenta dificultades para extraer patrones. Los estudiantes notan que no hay estructura por género ni frecuencia.  
    @Pregunta: ¿Qué evidencias indican que los datos están desorganizados? ¿Cómo podría aplicarse el modelado de un datamart para resolver esta situación?  
    @Respuesta Modelo: La desorganización impide filtrar por género o frecuencia. Un datamart permitiría estructurar por dimensiones, facilitando análisis y toma de decisiones.  
    @Conceptos Clave: Modelado de un datamart/datawarehouse, segmentación de datos, estructura dimensional.

    ## Datos curriculares
    - Competencia: {competencia}  
    - Capacidad: {capacidad}  
    - Criterio: {criterio}  
    - Complejidad: {complejidad}

    ### Temas Clave
    {temas_formateados}

    ### Caso:
    {caso}

    ## Instrucciones específicas

    1. Inicie con un solo `@Nombre` general para toda la ruta.

    2. Genere cinco retos, uno por cada etapa, usando el formato anterior.

    3. Alinee cada reto con la habilidad evaluada por etapa:

    - **Etapa 3 - Identificación del problema central**  
    - **Etapa 4 - Análisis causal y diagnóstico**  
    - **Etapa 5 - Generación de alternativas**  
    - **Etapa 6 - Evaluación y selección**  
    - **Etapa 7 - Plan de acción**

    4. Nivel de complejidad:
    - “Fácil” → Aplicación directa de conceptos.
    - “Difícil” → Interpretación, integración, hipótesis y decisión bajo incertidumbre.

    5. Evite repetir ideas o contextos entre retos.

    6. Use lenguaje técnico, académico y directo.

    7. Entregue los cinco retos juntos, sin explicaciones adicionales ni etiquetas nuevas.

    8. Verifique que cada reto tenga **exactamente** estas secciones:  
    `@Reto`, `@Contexto`, `@Pregunta`, `@Respuesta Modelo`, `@Conceptos Clave`.

    9. **Trazabilidad obligatoria**:  
    - La **subpregunta** debe exigir aplicar directamente un tema de `{temas_formateados}`.  
    - Ese mismo tema debe aparecer como **primer concepto en `@Conceptos Clave`**, sin reformulaciones ni sinónimos.  
    - Esto asegura la coherencia evaluativa entre la subpregunta y los conceptos que se espera que el estudiante aplique.
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

def lambda_handler(event, context):
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        required_fields = ["UsuarioId", "SilaboId", "UnidadId", "SesionId", "NombreCurso", "Competencia", "Capacidad", "Criterio", "Complejidad", "Temas", "Caso"]
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
        complejidad = body["Complejidad"]
        temas = body.get("Temas", None)
        caso = body["Caso"]

        prompt = RUTA_PROMPT.format(
            competencia=competencia,
            capacidad=capacidad,
            criterio=criterio,
            complejidad=complejidad,
            temas_formateados=', '.join(temas),
            caso=caso
        )

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