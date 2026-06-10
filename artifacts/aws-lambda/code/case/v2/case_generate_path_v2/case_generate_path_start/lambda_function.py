import json
import os
from typing import Any, Dict
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

CASE_GENERATE_PATH_WORKER_LAMBDA = os.environ["CASE_GENERATE_PATH_WORKER_LAMBDA"]

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
        "Complejidad",
        "NumeroRetos",
        "PlantillaId",
        "Competencias",
        "Caso",
    ]

    missing_fields = [f for f in required_fields if f not in body]

    if missing_fields:
        return f"Campos requeridos faltantes: " f"{missing_fields}"
    
    numero_retos = body.get("NumeroRetos", 0)

    if not isinstance(numero_retos, int) or numero_retos < 0:
        return "El campo 'NumeroRetos' debe ser un entero mayor o igual a 0."

    if numero_retos > 5:
        return "El campo 'NumeroRetos' no puede ser mayor a 5."

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
                        "El campo "
                        "'CamposTematicos' dentro "
                        "de cada criterio debe ser "
                        "una lista no vacía."
                    )

                for campo in campos:

                    if not all(
                        key in campo for key in ("CampoTematicoId", "CampoTematico")
                    ):
                        return (
                            "Cada campo temático debe "
                            "contener "
                            "'CampoTematicoId' y "
                            "'CampoTematico'."
                        )

    return None


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────


def lambda_handler(event: Dict[str, Any], context: Any):

    try:

        logger.info(f"Event case generate path start: {event}")

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
            FunctionName=CASE_GENERATE_PATH_WORKER_LAMBDA,
            InvocationType="Event",
            Payload=json.dumps(payload),
        )

        return {
            "statusCode": 202,
            "body": json.dumps(
                {
                    "success": True,
                    "message": "La generación de la ruta de aprendizaje ha sido iniciada.",
                    "history_id": history_id
                }
            ),
        }

    except Exception as e:

        logger.error(f"Error in case generate path start: {e}", exc_info=True)

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "success": False,
                    "message": "Error al iniciar la generación de la ruta de aprendizaje.",
                }
            ),
        }