import json
import os
from uuid import uuid4

import boto3

from aje_libs.common.logger import custom_logger

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ENVIRONMENT = os.environ["ENVIRONMENT"]
ENTERPRISE = os.environ["ENTERPRISE"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
OWNER = os.environ["OWNER"]

CASE_ENRICHMENT_CHALLENGE_WORKER_LAMBDA = os.environ["CASE_ENRICHMENT_CHALLENGE_WORKER_LAMBDA"]

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

lambda_client = boto3.client("lambda")

# ─────────────────────────────────────────────
# VALIDACIONES
# ─────────────────────────────────────────────


def _validate_body(body: dict):

    required_fields = [
        "UsuarioId",
        "SilaboId",
        "UnidadId",
        "SesionId",
        "NombreCurso",
        "Competencias",
        "Caso",
        "Retos",
    ]

    missing_fields = [f for f in required_fields if f not in body]

    if missing_fields:
        return f"Campos requeridos faltantes: " f"{missing_fields}"

    competencias = body.get("Competencias", [])

    if not isinstance(competencias, list) or len(competencias) == 0:
        return "El campo 'Competencias' debe ser una lista no vacía."
    
    for competencia in competencias:

        if not all(
            key in competencia
            for key in ("CompetenciaId", "Competencia", "Capacidades")
        ):
            return (
                "Cada competencia debe contener "
                "'CompetenciaId', "
                "'Competencia' y "
                "'Capacidades'."
            )

        capacidades = competencia.get("Capacidades", [])

        if not isinstance(capacidades, list) or len(capacidades) == 0:
            return (
                "El campo 'Capacidades' dentro "
                "de cada competencia debe ser "
                "una lista no vacía."
            )

        for capacidad in capacidades:

            if not all(
                key in capacidad for key in ("CapacidadId", "Capacidad", "Criterios")
            ):
                return (
                    "Cada capacidad debe contener "
                    "'CapacidadId', "
                    "'Capacidad' y "
                    "'Criterios'."
                )

            criterios = capacidad.get("Criterios", [])

            if not isinstance(criterios, list) or len(criterios) == 0:
                return (
                    "El campo 'Criterios' dentro "
                    "de cada capacidad debe ser "
                    "una lista no vacía."
                )

            for criterio in criterios:

                if not all(
                    key in criterio
                    for key in ("CriterioId", "Criterio", "CamposTematicos")
                ):
                    return (
                        "Cada criterio debe contener "
                        "'CriterioId', "
                        "'Criterio' y "
                        "'CamposTematicos'."
                    )

                campos = criterio.get("CamposTematicos", [])

                if not isinstance(campos, list) or len(campos) == 0:
                    return (
                        "El campo 'CamposTematicos' "
                        "dentro de cada criterio "
                        "debe ser una lista no vacía."
                    )

                for campo in campos:

                    if not all(
                        key in campo for key in ("CampoTematicoId", "CampoTematico")
                    ):
                        return (
                            "Cada campo temático "
                            "debe contener "
                            "'CampoTematicoId' "
                            "y 'CampoTematico'."
                        )

    retos = body.get("Retos", [])

    if not isinstance(retos, list) or len(retos) == 0:
        return "El campo 'Retos' debe ser una lista no vacía."

    for reto in retos:

        required_reto_fields = ["RetoId", "Pregunta", "RespuestaModelo"]

        missing_reto_fields = [f for f in required_reto_fields if f not in reto]

        if missing_reto_fields:
            return f"Faltan campos en reto: " f"{missing_reto_fields}"

    return None


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────


def lambda_handler(event, context):

    try:

        logger.info(f"Event enrichment challenge start: {event}")

        body = event.get("body", event)

        if isinstance(body, str):
            body = json.loads(body)

        error_message = _validate_body(body)

        if error_message:

            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": error_message}),
            }
        
        history_id = str(uuid4())
        payload = {
            **body,
            "history_id": history_id
        }

        lambda_client.invoke(
            FunctionName=CASE_ENRICHMENT_CHALLENGE_WORKER_LAMBDA,
            InvocationType="Event",
            Payload=json.dumps(payload),
        )

        return {
            "statusCode": 202,
            "body": json.dumps(
                {
                    "success": True,
                    "message": "Proceso de enriquecimiento de retos iniciado.",
                    "history_id": history_id
                }
            ),
        }

    except Exception as e:

        logger.error(f"Error in enrichment challenge start: {e}", exc_info=True)

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "success": False,
                    "message": "Error al iniciar el proceso de enriquecimiento de retos.",
                }
            ),
        }
