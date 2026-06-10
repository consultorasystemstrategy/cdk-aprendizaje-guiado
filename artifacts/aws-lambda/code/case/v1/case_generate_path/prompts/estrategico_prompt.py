"""
Prompt estratégico para generar rutas de retos formativos en el método del caso.
"""

RUTA_ESTRATEGICO_PROMPT = """
## Objetivo
Generar **cinco retos formativos secuenciales** basados en el caso proporcionado y los datos curriculares.
Cada reto debe corresponder a una **etapa del análisis del caso**, evaluando una habilidad específica por etapa.
El resultado servirá como una **ruta formativa completa**, coherente y contextualizada.

---

## Formato obligatorio
El modelo debe usar exactamente el siguiente formato:

@Nombre: [Título general de la ruta, máximo 6 palabras. Incluya al menos una palabra clave de los temas clave.]

(Después, por cada reto:)

@Reto: [Título breve del desafío]
@Contexto: [Describa el contexto del reto usando hechos, actores o tensiones del caso. El reto 5 debe incluir decisiones tomadas, actores clave y plazos.]
@Pregunta: [Una pregunta principal contextualizada + una subpregunta que aplique **uno de los temas clave** listados en {temas_formateados}.]
@Respuesta Modelo: [Respuesta analítica, contextual y coherente con la pregunta.]
@Conceptos Clave: [**Debe iniciar con el mismo tema clave exacto usado en la subpregunta**, seguido de otros conceptos o herramientas relacionados. Separe por comas y termine con punto.]

---

## Ejemplo de formato

@Nombre: Explorando patrones en lectura escolar

@Reto: Diagnóstico del caos en datos
@Contexto: La bibliotecaria enfrenta dificultades para extraer patrones. Los estudiantes notan que no hay estructura por género ni frecuencia.
@Pregunta: ¿Qué evidencias indican que los datos están desorganizados? ¿Cómo podría aplicarse el modelado de un datamart para resolver esta situación?
@Respuesta Modelo: La desorganización impide filtrar por género o frecuencia. Un datamart permitiría estructurar por dimensiones, facilitando análisis y toma de decisiones.
@Conceptos Clave: Modelado de un datamart/datawarehouse, segmentación de datos, estructura dimensional.

---

## Datos curriculares
- Competencia: {competencia}
- Capacidad: {capacidad}
- Criterio: {criterio}
- Complejidad: {complejidad}

### Temas clave
{temas_formateados}

### Caso base
{caso}

## Instrucciones específicas

1. Inicie con un único `@Nombre` general que represente toda la ruta.
2. Genere **cinco retos**, uno por cada etapa, usando el formato indicado.
3. Alinee cada reto con la habilidad evaluada por etapa:

    - **Etapa 3 - Identificación del problema central**
    - **Etapa 4 - Análisis causal y diagnóstico**
    - **Etapa 5 - Generación de alternativas**
    - **Etapa 6 - Evaluación y selección**
    - **Etapa 7 - Plan de acción**

4. Nivel de complejidad:
- “Fácil” → Aplicación directa de conceptos.
- “Intermedio / Difícil” → Interpretación, integración, hipótesis y decisión bajo incertidumbre.

5. Evite repetir ideas o contextos entre retos.
6. Use lenguaje técnico, académico y preciso.
7. Devuelva los cinco retos juntos, sin explicaciones adicionales ni etiquetas nuevas.
8. Verifique que cada reto tenga **exactamente** estas secciones:
`@Reto`, `@Contexto`, `@Pregunta`, `@Respuesta Modelo`, `@Conceptos Clave`.
9. **Trazabilidad obligatoria:**
   - La **subpregunta** debe aplicar directamente un tema de `{temas_formateados}`.
   - Ese mismo tema debe aparecer como **primer concepto** en `@Conceptos Clave`, **sin reformulaciones**.
   - Esto garantiza coherencia entre la evaluación y el contenido.
"""