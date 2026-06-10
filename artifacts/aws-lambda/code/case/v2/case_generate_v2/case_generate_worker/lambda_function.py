import json
import os
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict

from aje_libs.common.helpers.bedrock_helper import BedrockHelper
from aje_libs.common.helpers.dynamodb_helper import DynamoDBHelper
from aje_libs.common.helpers.ssm_helper import SSMParameterHelper
from aje_libs.common.logger import custom_logger

from .prompts.abp_prompt import ABP_PROMPT
from .prompts.clinico_psicologia_prompt import CLINICO_PSICOLOGIA_PROMPT
from .prompts.estrategico_prompt import CASO_ESTRATEGICO_PROMPT

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ENVIRONMENT = os.environ["ENVIRONMENT"]
ENTERPRISE = os.environ["ENTERPRISE"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
OWNER = os.environ["OWNER"]

CASE_HISTORY_TABLE = os.environ["CASE_HISTORY_TABLE"]

# ─────────────────────────────────────────────
# PARAMETER STORE
# ─────────────────────────────────────────────

ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/agent")

PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())

LLM_MODEL_ID = PARAMETER_VALUE["LLM_MODEL_ID"]
LLM_REGION = PARAMETER_VALUE["LLM_REGION"]
LLM_MAX_TOKENS = int(PARAMETER_VALUE["LLM_MAX_TOKENS"])

ssm_endpoints = SSMParameterHelper(
    f"/{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/endpoints"
)

ENDPOINTS = json.loads(ssm_endpoints.get_parameter_value())

URL_SAVE_CASE = ENDPOINTS["URL_SAVE_CASE"]

ssm_secret = SSMParameterHelper(f"/{ENVIRONMENT}/shared/internal-api-secret-key")

INTERNAL_API_SECRET_KEY = ssm_secret.get_parameter_value()

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

# ─────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────

case_history_table_helper = DynamoDBHelper(
    table_name=CASE_HISTORY_TABLE, pk_name="usuario_id", sk_name="date_time"
)

bedrock_helper = BedrockHelper(region_name=LLM_REGION)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _format_competencias(competencias: list) -> str:

    lines = []

    for comp in competencias:

        lines.append(
            f"Competencia [{comp['CompetenciaId']}]: " f"{comp['Competencia']}"
        )

        for cap in comp.get("Capacidades", []):

            lines.append(f"  Capacidad [{cap['CapacidadId']}]: " f"{cap['Capacidad']}")

            for crit in cap.get("Criterios", []):

                lines.append(
                    f"    Criterio [{crit['CriterioId']}]: " f"{crit['Criterio']}"
                )

                for campo in crit.get("CamposTematicos", []):

                    lines.append(
                        f"      Campo Temático "
                        f"[{campo['CampoTematicoId']}]: "
                        f"{campo['CampoTematico']}"
                    )

    return "\n".join(lines)


def _invoke_prompt(prompt: str, max_tokens: int, temperature: float = 0.7):

    response = bedrock_helper.converse(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        parameters={"max_tokens": max_tokens, "temperature": temperature, "top_p": 0.8},
    )

    logger.info(f"Respuesta del modelo: {response}")

    return response


def _save_history(
    history_id: str,
    usuario_id: int | None,
    silabo_id: int | None,
    unidad_id: int | None,
    sesion_id: int | None,
    plantilla_id: int | None,
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
        "silabo_id": silabo_id,
        "unidad_id": unidad_id,
        "sesion_id": sesion_id,
        "plantilla_id": plantilla_id,
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

    case_history_table_helper.put_item(data=item)


def _http_json(
    method: str, url: str, headers: dict, payload: dict | None = None, timeout: int = 20
) -> dict:
    data = None
    req_headers = dict(headers or {})

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:

            body = resp.read().decode("utf-8")

            return {
                "status_code": resp.status,
                "body": json.loads(body) if body else None,
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


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────


def lambda_handler(event: Dict[str, Any], context: Any):

    logger.info(f"Event case generate worker: {event}")

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
    plantilla_id = body.get("PlantillaId")

    headers = {"X-Web-Api-Secret": INTERNAL_API_SECRET_KEY}

    prompt = None
    case = None
    raw_response = None

    input_tokens = 0
    output_tokens = 0

    status = "failed"

    error_message = None
    web_api_response = None

    # ── Lógica IA ─────────────────────────────────────────────────────────────
    try:
        nombre_curso = body["NombreCurso"]
        complejidad = body["Complejidad"]
        contexto = body["Contexto"]
        competencias = body["Competencias"]

        competencias_formateadas = _format_competencias(competencias)

        PROMPT_MAP = {
            0: CASO_ESTRATEGICO_PROMPT,
            1: ABP_PROMPT,
            2: CLINICO_PSICOLOGIA_PROMPT,
        }

        prompt_template = PROMPT_MAP.get(plantilla_id)

        if prompt_template is None:
            raise ValueError(f"PlantillaId '{plantilla_id}' no válido.")

        prompt = prompt_template.format(
            nombre_curso=nombre_curso,
            competencias_formateadas=competencias_formateadas,
            complejidad=complejidad,
            contexto=contexto,
        )

        raw_response = _invoke_prompt(
            prompt=prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.7
        )
        case = raw_response["output"]["message"]["content"][0]["text"]
        input_tokens = raw_response["usage"]["inputTokens"]
        output_tokens = raw_response["usage"]["outputTokens"]

        status = "completed"

    except Exception as e:
        error_message = str(e)
        logger.error(
            f"Error in case generate worker. "
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
            "plantilla_id": plantilla_id,
            "status": status,
            "case": case,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **({"error": {"message": error_message}} if error_message else {}),
        }

        logger.info(
            f"Enviando resultado al API. history_id={history_id}, usuario_id={usuario_id}, status={status}"
        )

        web_api_response = _http_json(
            method="POST",
            url=URL_SAVE_CASE,
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
            f"Error al llamar al API. history_id={history_id}, usuario_id={usuario_id}: {api_error}",
            exc_info=True,
        )

    _save_history(
        history_id=history_id,
        usuario_id=usuario_id,
        silabo_id=silabo_id,
        unidad_id=unidad_id,
        sesion_id=sesion_id,
        plantilla_id=plantilla_id,
        status=status,
        prompt=prompt,
        ai_result=case,
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
                "message": "Proceso de generación de caso finalizado.",
                "history_id": history_id,
            }
        ),
    }