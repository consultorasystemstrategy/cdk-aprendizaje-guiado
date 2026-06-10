import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict

from aje_libs.bd.helpers.pinecone_helper import PineconeHelper
from aje_libs.common.helpers.bedrock_helper import BedrockHelper
from aje_libs.common.helpers.dynamodb_helper import DynamoDBHelper
from aje_libs.common.helpers.secrets_helper import SecretsHelper
from aje_libs.common.helpers.ssm_helper import SSMParameterHelper
from aje_libs.common.logger import custom_logger

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ENVIRONMENT = os.environ["ENVIRONMENT"]
ENTERPRISE = os.environ["ENTERPRISE"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
OWNER = os.environ["OWNER"]

LEARNING_PATH_HISTORY_TABLE = os.environ["LEARNING_PATH_HISTORY_TABLE"]

# ─────────────────────────────────────────────
# PARAMETER STORE
# ─────────────────────────────────────────────

ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/agent")

PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())

EMBEDDING_MODEL_ID = PARAMETER_VALUE["EMBEDDING_MODEL_ID"]
EMBEDDING_REGION = PARAMETER_VALUE["EMBEDDING_REGION"]

LLM_MODEL_ID = PARAMETER_VALUE["LLM_MODEL_ID"]
LLM_REGION = PARAMETER_VALUE["LLM_REGION"]
LLM_MAX_TOKENS = int(PARAMETER_VALUE["LLM_MAX_TOKENS"])

PINECONE_MAX_RETRIEVE_DOCUMENTS = int(
    PARAMETER_VALUE["PINECONE_MAX_RETRIEVE_DOCUMENTS"]
)

PINECONE_MIN_THRESHOLD = float(PARAMETER_VALUE["PINECONE_MIN_THRESHOLD"])

ssm_endpoints = SSMParameterHelper(
    f"/{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/endpoints"
)

ENDPOINTS = json.loads(ssm_endpoints.get_parameter_value())

URL_SAVE_LEARNING_PATH = ENDPOINTS["URL_SAVE_LEARNING_PATH"]

ssm_secret = SSMParameterHelper(f"/{ENVIRONMENT}/shared/internal-api-secret-key")

INTERNAL_API_SECRET_KEY = ssm_secret.get_parameter_value()

# ─────────────────────────────────────────────
# SECRETS
# ─────────────────────────────────────────────

secret_pinecone = SecretsHelper(
    f"{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/pinecone-api"
)

PINECONE_INDEX_NAME = secret_pinecone.get_secret_value("PINECONE_INDEX_NAME")

PINECONE_API_KEY = secret_pinecone.get_secret_value("PINECONE_API_KEY")

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

# ─────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────

learning_path_table_helper = DynamoDBHelper(
    table_name=LEARNING_PATH_HISTORY_TABLE, pk_name="usuario_id", sk_name="date_time"
)

pinecone_helper = PineconeHelper(
    index_name=PINECONE_INDEX_NAME,
    api_key=PINECONE_API_KEY,
    embeddings_model_id=EMBEDDING_MODEL_ID,
    embeddings_region=EMBEDDING_REGION,
    max_retrieve_documents=(PINECONE_MAX_RETRIEVE_DOCUMENTS),
    min_threshold=PINECONE_MIN_THRESHOLD,
)

bedrock_helper = BedrockHelper(region_name=LLM_REGION)

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

PATH_PROMPT = """
### Instrucción
Eres un experto en diseño pedagógico. Genera una ruta de aprendizaje con exactamente {numero_retos} retos.
Cada reto debe integrar y relacionar múltiples criterios de aprendizaje y campos temáticos de forma cohesionada.

### Estructura pedagógica del curso:
{competencias_formateadas}

### Documentación de referencia:
[{context}]

### Nivel de complejidad: {complejidad}

### Reglas de generación:
- Genera exactamente {numero_retos} retos. Ni más, ni menos.
- Cada reto DEBE cubrir criterios y campos temáticos de distintos niveles jerárquicos cuando sea posible.
- El reto debe ser integrador: su pregunta debe exigir al estudiante relacionar conceptos de varios criterios o temas.
- Distribuye la cobertura de criterios de forma que el conjunto de retos abarque todos los criterios disponibles.
- Usa la documentación de referencia para enriquecer las respuestas modelo cuando sea relevante.
- Si no hay documentación de referencia, basa los retos en tu conocimiento general del tema.
- Por cada criterio seleccionado en un reto, debes indicar ÚNICAMENTE los CampoTematicoId (de cualquier nivel)
  que ese reto trabaja dentro de ese criterio. No mezcles campos de distintos criterios en la misma lista.

### Formato de respuesta
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques de código markdown, sin explicaciones.
El JSON debe tener exactamente esta estructura:

{{
  "nombre": "Frase en español con 3 a 6 palabras separadas por espacios, que describa el eje temático de la ruta. No uses CamelCase, guiones ni caracteres especiales. Evita frases genéricas.",
  "retos": [
    {{
      "titulo": "Frase en español con 3 a 6 palabras separadas por espacios que describa el reto. No uses CamelCase ni guiones.",
      "criterios": [
        {{
          "criterio_id": <CriterioId (entero)>,
          "campos_tematicos_ids": [<lista de CampoTematicoId (enteros) de ese criterio que cubre este reto>]
        }}
      ],
      "pregunta": "Pregunta abierta e integradora que relacione los criterios y temas cubiertos, nivel {complejidad}",
      "respuesta_modelo": "Respuesta clara, estructurada y completa que sirva como guía de evaluación"
    }}
  ]
}}
}}
"""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _format_competencias(competencias: list) -> str:

    lines = []

    for comp in competencias:

        lines.append(
            f"Competencia " f"[{comp['CompetenciaId']}]: " f"{comp['Competencia']}"
        )

        for cap in comp.get("Capacidades", []):

            lines.append(
                f"  Capacidad " f"[{cap['CapacidadId']}]: " f"{cap['Capacidad']}"
            )

            for crit in cap.get("Criterios", []):

                lines.append(
                    f"    Criterio " f"[{crit['CriterioId']}]: " f"{crit['Criterio']}"
                )

                for campo in crit.get("CamposTematicos", []):

                    lines.append(
                        f"      Campo Temático "
                        f"[{campo['CampoTematicoId']}]: "
                        f"{campo['CampoTematico']}"
                    )

    return "\n".join(lines)


def _build_query_text(competencias: list) -> str:

    criterios_texts = []
    temas_texts = []

    for comp in competencias:

        for cap in comp.get("Capacidades", []):

            for crit in cap.get("Criterios", []):

                criterios_texts.append(crit["Criterio"])

                for campo in crit.get("CamposTematicos", []):

                    temas_texts.append(campo["CampoTematico"])

    query_parts = []

    if criterios_texts:

        query_parts.append("Criterios: " + ". ".join(criterios_texts))

    if temas_texts:

        query_parts.append("Temas: " + ", ".join(temas_texts))

    return ". ".join(query_parts)


def _parse_llm_response(raw_text: str) -> dict:

    clean = raw_text.strip()

    clean = clean.removeprefix("```json")
    clean = clean.removeprefix("```")
    clean = clean.removesuffix("```")
    clean = clean.strip()

    return json.loads(clean)


def _invoke_prompt(prompt: str, max_tokens: int, temperature: float = 0.7):

    response = bedrock_helper.converse(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        parameters={"max_tokens": max_tokens, "temperature": temperature, "top_p": 0.2},
    )

    return response


def _get_documents_context(question: str, resources: list) -> str:

    try:

        filter_conditions = {}

        if resources:

            if isinstance(resources, str):
                resources = resources.split(",")

            resource_ids = [str(rid) for rid in resources]

            filter_conditions["resource_id"] = {"$in": resource_ids}

        relevant_data = pinecone_helper.search_by_text(
            query_text=question,
            filter_conditions=(filter_conditions if filter_conditions else None),
            return_format="text",
            text_field="text",
        )

        return relevant_data

    except Exception as e:

        logger.error(f"Error al consultar Pinecone: {e}")

        return ""


def _retrieve_context(query_text: str, resources: list) -> str:

    if not resources:

        return (
            "No se cuenta con material documental. "
            "Genera los retos únicamente con base "
            "en tu conocimiento general sobre el tema."
        )

    return _get_documents_context(query_text, resources)


def _http_json(
    method: str, url: str, headers: dict, payload: dict | None = None, timeout: int = 15
) -> dict:

    data = None

    req_headers = dict(headers or {})

    if payload is not None:

        data = json.dumps(payload).encode("utf-8")

        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=req_headers, method=method)

    try:

        with urllib.request.urlopen(req, timeout=timeout) as resp:

            body = resp.read().decode("utf-8")

            return {
                "status_code": resp.status,
                "body": (json.loads(body) if body else None),
            }

    except urllib.error.HTTPError as e:

        error_body = e.read().decode("utf-8")

        logger.error(
            f"HTTPError {e.code} calling external API " f"- Body: {error_body}"
        )

        raise

    except Exception as e:

        logger.error(f"Error calling external API: {e}", exc_info=True)

        raise


def _save_history(
    history_id: str,
    usuario_id: int | None,
    silabo_id: int | None,
    unidad_id: int | None,
    sesion_id: int | None,
    status: str,
    prompt: str | None,
    ai_result: str | None,
    raw_response: dict | None,
    input_tokens: int,
    output_tokens: int,
    web_api_response: dict | None,
    error_message: str | None = None,
):

    ttl_timestamp = int((datetime.now() + timedelta(days=5)).timestamp())

    item = {
        "history_id": history_id,
        "usuario_id": usuario_id,
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo_metodo_id": 674,
        "silabo_id": silabo_id,
        "unidad_id": unidad_id,
        "sesion_id": sesion_id,
        "status": status,
        "prompt": prompt,
        "ai_result": ai_result,
        "raw_response": (
            json.dumps(raw_response, ensure_ascii=False) if raw_response else None
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "web_api_response": (
            json.dumps(web_api_response, ensure_ascii=False)
            if web_api_response
            else None
        ),
        "error_message": error_message,
        "ttl": ttl_timestamp,
    }

    learning_path_table_helper.put_item(data=item)


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────


def lambda_handler(event: Dict[str, Any], context: Any):

    logger.info(f"Event path generate worker: {event}")

    body = event

    if isinstance(body, str):
        body = json.loads(body)

    history_id = body.get("history_id")
    if not history_id:
        raise ValueError("El campo 'history_id' es requerido en el payload.")
    
    usuario_id = body.get("UsuarioId")
    silabo_id = body.get("SilaboId")
    unidad_id = body.get("UnidadId")
    sesion_id = body.get("SesionId")

    headers = {"X-Web-Api-Secret": INTERNAL_API_SECRET_KEY}

    prompt = None
    raw_result = None
    learning_path = None
    raw_response = None

    input_tokens = 0
    output_tokens = 0

    status = "failed"

    error_message = None
    web_api_response = None

    # ── Lógica IA ─────────────────────────────────────────────────────────────
    try:

        competencias = body["Competencias"]
        resources = body.get("ResourcesIds", None)

        complejidad = body["Complejidad"]
        numero_retos = body["NumeroRetos"]

        competencias_formateadas = _format_competencias(competencias)

        query_text = _build_query_text(competencias)

        pinecone_context = _retrieve_context(query_text, resources)

        prompt = PATH_PROMPT.format(
            competencias_formateadas=(competencias_formateadas),
            complejidad=complejidad,
            numero_retos=numero_retos,
            context=pinecone_context,
        )

        raw_response = _invoke_prompt(
            prompt=prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.7
        )
        raw_result = raw_response["output"]["message"]["content"][0]["text"]
        input_tokens = raw_response["usage"]["inputTokens"]
        output_tokens = raw_response["usage"]["outputTokens"]

        try:
            learning_path = _parse_llm_response(raw_result)
        except json.JSONDecodeError:
            raise ValueError("Respuesta del modelo no es un JSON válido.")

        status = "completed"

    except Exception as e:
        error_message = str(e)
        logger.error(
            f"Error in path generate worker. "
            f"history_id={history_id}, "
            f"usuario_id={usuario_id}: "
            f"{error_message}",
            exc_info=True,
        )

    # ── Llamada al API (siempre se ejecuta, éxito o fallo de la IA) ──────────
    try:

        save_payload = {
            "history_id": history_id,
            "usuario_id": usuario_id,
            "silabo_id": silabo_id,
            "unidad_id": unidad_id,
            "sesion_id": sesion_id,
            "status": status,
            "learning_path": learning_path,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **({"error": {"message": error_message}} if error_message else {}),
        }

        logger.info(
            f"Enviando resultado al API. history_id={history_id}, usuario_id={usuario_id}, status={status}"
        )

        web_api_response = _http_json(
            method="POST",
            url=URL_SAVE_LEARNING_PATH,
            headers=headers,
            payload=save_payload,
            timeout=15,
        )

        logger.info(
            f"Respuesta del API. history_id={history_id}, usuario_id={usuario_id}: {web_api_response}"
        )

    except Exception as api_error:
        web_api_response = {"error": str(api_error)}
        logger.error(
            f"Error al llamar al API. "
            f"history_id={history_id}, "
            f"usuario_id={usuario_id}: "
            f"{api_error}",
            exc_info=True,
        )

    # ── HISTORIAL ────────────────────────────

    _save_history(
        history_id=history_id,
        usuario_id=usuario_id,
        silabo_id=silabo_id,
        unidad_id=unidad_id,
        sesion_id=sesion_id,
        status=status,
        prompt=prompt,
        ai_result=raw_result,
        raw_response=raw_response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        web_api_response=web_api_response,
        error_message=error_message,
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "success": True,
                "message": "Proceso de generación de ruta de aprendizaje finalizado.",
                "history_id": history_id,
            }
        ),
    }