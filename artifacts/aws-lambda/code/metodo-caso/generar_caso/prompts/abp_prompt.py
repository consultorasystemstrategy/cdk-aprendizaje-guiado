"""
Prompt template para generar proyectos ABP (Aprendizaje Basado en Proyectos)

"""

ABP_PROMPT = """
## Objetivo
Genera un texto narrativo completo, coherente, continuo y argumentativo que describa un proyecto ABP (Aprendizaje Basado en Proyectos) estructurado en cinco retos:
1. Análisis del contexto.
2. Definición de necesidades.
3. Propuesta de solución.
4. Evaluación de riesgos.
5. Validación final.

El resultado debe ser un texto fluido, sin encabezados visibles ni listas, pero que internamente siga la secuencia y estructura lógica de los cinco retos.

---

## Datos curriculares
- Curso: {nombre_curso}
- Competencia: {competencia}
- Capacidad: {capacidad}
- Criterio: {criterio}
- Nivel de complejidad: {complejidad}
- Temas clave: {temas_formateados}

---

## Texto base (opcional, puede estar vacío)
{contexto}

Si el texto base está presente, extrae y reinterpreta:
- Nombre de la empresa y sector.
- Dilema o problema principal.
- Propósito o misión institucional.
- Datos relevantes (cifras, contexto, actores, tensiones).
Usa esta información para reconstruir un proyecto coherente y original, sin copiar texto literal.

Si no hay texto base, genera un proyecto nuevo coherente con los datos curriculares y el nivel de complejidad **{complejidad}**.
El dilema o desafío debe centrarse en los **temas clave** (**{temas_formateados}**) e implicar la competencia, capacidad y criterio proporcionados, sin mencionarlos directamente.

---

## Instrucciones de redacción

1. **Título del proyecto**
   - Antes de iniciar el texto, genera un título representativo (8-14 palabras) que resuma el propósito del aprendizaje y el desafío central del proyecto.
   - Debe estar alineado con el tema y la habilidad, ser inspirador, claro y contextual.
   - Ejemplo: “Clustering para personalizar la atención y recuperar la confianza del paciente”.
   - El título debe colocarse al inicio del texto final.

2. **Nivel de complejidad**
   - Si el nivel es **fácil**, organiza el texto por títulos de retos (sin subtítulos, listas ni viñetas).
   - Si el nivel es **intermedio o difícil**, redacta el texto en párrafos continuos (sin títulos ni listas), con lenguaje profesional, claro y motivador.

3. **Estilo narrativo**
   - Lenguaje profesional, claro y realista, accesible a estudiantes universitarios.
   - Mantén tono analítico, propositivo y didáctico.
   - Transiciones suaves entre los retos (sin separadores visuales).
   - No incluyas los datos curriculares ni instrucciones en el texto final.

---

## Estructura esperada del texto

El texto debe integrar los cinco retos de manera continua y coherente:

1. **Analiza el problema y su contexto:**
Objetivo: Contextualizar y explicar críticamente la situación.
Incluir:
    - Contexto institucional y sectorial.
    - Descripción del problema central y sus causas.
    - Al menos dos actores y su relación con el problema.
    - Datos simulados (indicadores, cifras) que reflejen magnitud.
    - Restricciones reales (tiempo, cultura, presupuesto, tecnología).
    - Rol profesional del estudiante.

2. **Identifica necesidades, requisitos y objetivos:**
Objetivo: Definir criterios de solución desde el análisis del contexto.
Incluir:
    - Necesidades concretas de los actores.
    - Requisitos técnicos, operativos y humanos.
    - Objetivos claros y medibles en infinitivo.

3. **Propone una solución justificada:**
Objetivo: Diseñar y argumentar una propuesta viable y coherente.
Incluir:
    - Descripción general de la solución (nombre, propósito, funcionamiento).
    - Componentes principales o fases.
    - Correspondencia con los requisitos previos.
    - Justificación de pertinencia, viabilidad y valor añadido.

4. **Evalúa riesgos y limitaciones:**
Objetivo: Anticipar y mitigar posibles obstáculos.
Incluir:
    - Identificación de riesgos técnicos, humanos, sociales o económicos.
    - Limitaciones del entorno institucional.
    - Estrategias realistas de mitigación.

5. **Validación y ajuste final:**
Objetivo: Reflexionar y mejorar la propuesta.
Incluir:
    - Supuestos de éxito.
    - Posibles fallos detectados.
    - Propuestas de mejora.
    - Valoración crítica del impacto y sostenibilidad.

---

## Reglas finales
- Extensión total: entre 600 y 1300 palabras.
- Párrafos de 4 a 10 líneas.
- Redacción continua, sin listas ni separadores visibles.
- Mantén coherencia entre problema, necesidades, solución, riesgos y reflexión.
- No incluyas notas meta ni explicaciones sobre los retos.
- Prioriza la claridad, el realismo y la tensión narrativa del proyecto.
"""