RUTA_DINAMICA_PROMPT = """
### Instrucción
Eres un experto en diseño pedagógico y evaluación basada en competencias.

Genera una ruta de aprendizaje con exactamente {numero_retos} retos integradores, basados en el caso proporcionado y alineados con la estructura pedagógica del curso.

Cada reto debe promover análisis, integración de conocimientos, razonamiento aplicado y toma de decisiones según el nivel de complejidad indicado.

### Estructura pedagógica del curso:
{competencias_formateadas}

### Caso base:
{caso}

### Nivel de complejidad:
{complejidad}

### Reglas de generación:
- Genera exactamente {numero_retos} retos.
- Cada reto debe integrar múltiples criterios y campos temáticos cuando sea posible.
- Distribuye la cobertura de criterios de manera equilibrada para que el conjunto de retos abarque toda la estructura pedagógica disponible.
- Evita repetir exactamente los mismos criterios o preguntas entre retos.
- Los retos deben evolucionar progresivamente en profundidad y complejidad.
- Cada reto debe derivarse directamente del caso proporcionado.
- No inventes información fuera del contexto del caso.
- Cada reto debe plantear una situación, análisis, problema o decisión diferente.
- Por cada criterio seleccionado en un reto, indica ÚNICAMENTE los CampoTematicoId que ese reto trabaja dentro de ese criterio.
- No mezcles campos temáticos de distintos criterios en la misma lista.
- Las preguntas deben ser abiertas, analíticas e integradoras.
- Las respuestas modelo deben ser claras, estructuradas y útiles para evaluación docente.

### Nivel de complejidad:
- "Fácil":
  Aplicación directa de conceptos, identificación de elementos clave y resolución guiada.

- "Intermedio":
  Relación entre conceptos, análisis contextual, interpretación y justificación de decisiones.

- "Difícil":
  Integración multidisciplinaria, razonamiento complejo, evaluación crítica, hipótesis y toma de decisiones bajo incertidumbre.

### Formato de respuesta
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques de código markdown, sin explicaciones.
El JSON debe tener exactamente esta estructura:

{{
  "nombre": "Frase en español con 3 a 6 palabras separadas por espacios, que describa el eje temático de la ruta. No uses CamelCase, guiones ni caracteres especiales.",
  "retos": [
    {{
      "titulo": "Frase en español con 3 a 6 palabras separadas por espacios que describa el reto.",
      "criterios": [
        {{
          "criterio_id": <CriterioId (entero)>,
          "campos_tematicos_ids": [<lista de CampoTematicoId (enteros)>]
        }}
      ],
      "contexto": "Fragmento o extracto relevante del caso que sirve como base del reto.",
      "pregunta": "Pregunta abierta, integradora y analítica alineada al nivel de complejidad.",
      "respuesta_modelo": "Respuesta clara, estructurada y técnicamente correcta que sirva como guía de evaluación."
    }}
  ]
}}
"""