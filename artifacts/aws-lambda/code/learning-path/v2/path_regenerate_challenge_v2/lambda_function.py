import json
import os
from datetime import datetime, timedelta

from aje_libs.common.helpers.bedrock_helper import BedrockHelper
from aje_libs.common.helpers.dynamodb_helper import DynamoDBHelper
from aje_libs.common.helpers.ssm_helper import SSMParameterHelper
from aje_libs.common.logger import custom_logger

# Configuración
ENVIRONMENT = os.environ["ENVIRONMENT"]
ENTERPRISE = os.environ["ENTERPRISE"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
OWNER = os.environ["OWNER"]
REGENERATED_HISTORY_TABLE = os.environ["REGENERATED_CHALLENGES_HISTORY_TABLE"]

# Parameter Store
ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/agent")
PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())
LLM_MODEL_ID = PARAMETER_VALUE["LLM_MODEL_ID"]
LLM_REGION = PARAMETER_VALUE["LLM_REGION"]
LLM_MAX_TOKENS = int(PARAMETER_VALUE["LLM_MAX_TOKENS"])

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

regenerated_table_helper = DynamoDBHelper(
    table_name=REGENERATED_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="date_time"
)

bedrock_helper = BedrockHelper(region_name=LLM_REGION)

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

REGENERATE_CHALLENGE_PROMPT = """
## Resumen de la tarea:
Eres un experto en pedagogía y en el curso {nombre_curso}. Tu tarea es regenerar un reto de aprendizaje existente aplicando las indicaciones del usuario, sin alterar los criterios de evaluación ni los campos temáticos asociados.

## Reto original:
- Título: {titulo_reto}
- Pregunta: {pregunta}
- Respuesta modelo: {respuesta_modelo}

## Criterios de evaluación (NO modificar su alcance):
{criterios_formateados}

## Indicaciones del usuario:
{indicaciones}

## Instrucciones para el modelo:
- APLICA las indicaciones del usuario para modificar el título, la pregunta y/o la respuesta modelo del reto.
- MANTÉN el reto alineado a los criterios de evaluación y campos temáticos proporcionados. Las indicaciones no pueden redirigir el reto hacia otros criterios o temas distintos a los listados.
- ASEGÚRATE de que la nueva pregunta sea abierta, clara y coherente con los criterios.
- REDACTA una respuesta modelo completa y estructurada que responda correctamente la nueva pregunta.
- USA un lenguaje claro, técnico y directo.

## Formato de respuesta:
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques de código markdown, sin explicaciones.
El JSON debe tener exactamente esta estructura:

{{
  "titulo": "Título breve del reto regenerado",
  "pregunta": "Pregunta regenerada abierta e integradora basada en los criterios",
  "respuesta_modelo": "Respuesta clara, estructurada y completa que sirva como guía de evaluación"
}}
"""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _format_criterios(criterios: list) -> str:
    """
    Formatea la lista de criterios con sus campos temáticos
    para incluirla en los prompts.
    """
    lines = []
    for crit in criterios:
        lines.append(f"- Criterio [{crit['CriterioId']}]: {crit['Criterio']}")
        for campo in crit.get("CamposTematicos", []):
            lines.append(f"  · [{campo['CampoTematicoId']}] {campo['CampoTematico']}")
    return "\n".join(lines)


def _parse_llm_response(
    raw_text: str
) -> dict:

    clean = raw_text.strip()

    clean = clean.removeprefix("```json")
    clean = clean.removeprefix("```")
    clean = clean.removesuffix("```")
    clean = clean.strip()

    return json.loads(clean)

def _invoke_prompt(prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
    response = bedrock_helper.converse(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        parameters={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.2
        }
    )
    logger.info(f"Respuesta del modelo: {response}")
    return response


def _upload_challenge(
    usuario_id: int,
    silabo_id: int,
    unidad_id: int,
    sesion_id: int,
    indicaciones: str,
    prompt: str,
    ai_result: str,
    input_tokens: int,
    output_tokens: int
):
    try:
        ttl_timestamp = int((datetime.now() + timedelta(seconds=432000)).timestamp())  # TTL 5 días

        item = {
            "usuario_id": usuario_id,
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "silabo_id": silabo_id,
            "unidad_id": unidad_id,
            "sesion_id": sesion_id,
            "indicaciones": indicaciones,
            "prompt": prompt,
            "ai_result": ai_result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttl": ttl_timestamp
        }

        regenerated_table_helper.put_item(data=item)
        logger.info(f"Reto regenerado persistido: usuario_id={usuario_id}, sesion_id={sesion_id}")

    except Exception as e:
        logger.error(f"Error al persistir reto regenerado: {e}")

# ─────────────────────────────────────────────
# VALIDACIONES
# ─────────────────────────────────────────────

def _validate_criterios(criterios: list) -> str | None:
    """
    Valida la lista de criterios con sus campos temáticos.
    Retorna un mensaje de error si falla, None si es válida.
    """
    if not isinstance(criterios, list) or len(criterios) == 0:
        return "El campo 'Criterios' debe ser una lista no vacía."

    for crit in criterios:
        if not all(k in crit for k in ("CriterioId", "Criterio", "CamposTematicos")):
            return "Cada criterio debe contener 'CriterioId', 'Criterio' y 'CamposTematicos'."

        campos = crit.get("CamposTematicos", [])
        if not isinstance(campos, list) or len(campos) == 0:
            return "El campo 'CamposTematicos' dentro de cada criterio debe ser una lista no vacía."

        for campo in campos:
            if not all(k in campo for k in ("CampoTematicoId", "CampoTematico")):
                return "Cada campo temático debe contener 'CampoTematicoId' y 'CampoTematico'."

    return None


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        logger.info(f"Event path regenerate: {event}")

        body = event.get("body")
        if not body:
            return {"statusCode": 400, "body": json.dumps({"success": False, "message": "Body requerido"})}
        body = json.loads(body) if isinstance(body, str) else body

        # Validar campos requeridos
        required_fields = [
            "UsuarioId", "SilaboId", "UnidadId", "SesionId",
            "NombreCurso", "Indicaciones", "Reto"
        ]
        missing_fields = [f for f in required_fields if f not in body]
        if missing_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": f"Campos requeridos faltantes: {missing_fields}"})
            }

        # Validar estructura de Reto
        reto = body["Reto"]
        required_reto_fields = ["Titulo", "Pregunta", "RespuestaModelo", "Criterios"]
        missing_reto_fields = [f for f in required_reto_fields if f not in reto]
        if missing_reto_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": f"Campos requeridos faltantes en 'Reto': {missing_reto_fields}"})
            }

        # Validar criterios dentro de Reto
        validation_error = _validate_criterios(reto["Criterios"])
        if validation_error:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": validation_error})
            }

        usuario_id   = body["UsuarioId"]
        silabo_id    = body["SilaboId"]
        unidad_id    = body["UnidadId"]
        sesion_id    = body["SesionId"]
        nombre_curso = body["NombreCurso"]
        indicaciones = body["Indicaciones"]

        titulo_reto      = reto["Titulo"]
        pregunta         = reto["Pregunta"]
        respuesta_modelo = reto["RespuestaModelo"]
        criterios        = reto["Criterios"]

        criterios_formateados = _format_criterios(criterios)

        prompt = REGENERATE_CHALLENGE_PROMPT.format(
            nombre_curso=nombre_curso,
            titulo_reto=titulo_reto,
            pregunta=pregunta,
            respuesta_modelo=respuesta_modelo,
            criterios_formateados=criterios_formateados,
            indicaciones=indicaciones,
        )

        response = _invoke_prompt(prompt=prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.7)
        raw_result = response["output"]["message"]["content"][0]["text"]
        input_tokens  = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]

        _upload_challenge(
            usuario_id=usuario_id,
            silabo_id=silabo_id,
            unidad_id=unidad_id,
            sesion_id=sesion_id,
            indicaciones=indicaciones,
            prompt=prompt,
            ai_result=raw_result,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        try:
            challenge = _parse_llm_response(raw_result)
        except json.JSONDecodeError:
            return {
                "statusCode": 502, # Bad Gateway, porque el error fue en la respuesta del modelo
                "body": json.dumps({
                    "success": False,
                    "message": "El modelo no retornó el reto en formato esperado."
                })
            }

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "reto": challenge,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        }

    except Exception as e:
        logger.error(f"Error in path regenerate: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"success": False, "message": str(e)})
        }