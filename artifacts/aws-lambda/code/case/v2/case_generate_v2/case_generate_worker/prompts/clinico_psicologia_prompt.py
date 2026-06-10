"""
Prompt template para generar casos clínicos de psicología

"""

CLINICO_PSICOLOGIA_PROMPT = """
## Objetivo
Genera un **caso clínico psicológico completo, realista y éticamente redactado**, siguiendo el **modelo de formulación del caso clínico (Case Formulation Model)** propuesto por la **APA (2020)** y **Eells (2010)**.  
El texto servirá como insumo para que el estudiante:
- Analice el caso clínico.  
- Practique el diagnóstico y diagnóstico diferencial.  
- Formule una conceptualización teórica.  
- Diseñe un plan de intervención.

---

## Datos curriculares
- Curso: {nombre_curso}
- Nivel de complejidad: {complejidad}

### Estructura pedagógica:
{competencias_formateadas}

---

## Texto base
{contexto}

Del texto base, extrae y reinterpreta con precisión los siguientes elementos:

- **Trastorno principal:** eje central del caso clínico.  
- **Comorbilidad secundaria (opcional):** condición asociada que complejiza el cuadro.  
- **Enfoque teórico orientador:** marco conceptual que guiará la descripción clínica.  
- **Contexto del caso:** entorno donde ocurre la atención psicológica (consulta, hospital, escuela, empresa, telepsicología, etc.).  
- **Edad, género y ocupación del paciente.**  
- **Estado civil y situación familiar.**  
- **Nivel educativo y cultural.**  
- **Duración aproximada del problema.**  
- **Factores psicosociales relevantes** (estresores, duelos, conflictos, trauma, aislamiento, etc.).  
- **Propósito pedagógico:** orientar el enfoque del caso (p. ej., diagnóstico diferencial, formulación del caso, diseño de intervención), **sin mencionarlo explícitamente** en la narrativa.

**Anonimiza** toda información identificable: use iniciales o seudónimo (p. ej. "M.R.").  
Si el texto base incluye ideación suicida o riesgo inminente, descríbalo claramente (nivel: bajo/moderado/alto) y señale la necesidad de derivación/contención urgente sin ofrecer instrucciones terapéuticas detalladas.
Usa toda esta información para **reconstruir un caso coherente, verosímil y original**, manteniendo correspondencia con los datos provistos, pero **sin copiar frases ni etiquetas del texto base**.
Verifica internamente que **cada dato del texto base** esté **implícitamente reflejado** en la narrativa final del caso (edad, género, contexto, duración, enfoque, etc.).

---

## Instrucciones de redacción
- Lenguaje técnico, descriptivo y objetivo.  
- Uso correcto de terminología clínica (afecto, juicio, insight, etc.).  
- Evitar tecnicismos excesivos o conclusiones prematuras.  
- Mantener coherencia entre las secciones (motivo, contexto, observaciones).  
- No usar listas dentro del cuerpo narrativo (solo en encabezados o apartados técnicos).  
- Mantener estructura visual jerárquica, como informe institucional.  
- **Las palabras literales del paciente deben ir entre comillas y en cursiva** (“*…*”). No use cursivas ni comillas fuera de ese contexto.

---

## Estructura esperada del texto

1. **Presentación del caso (100-200 palabras):**
Incluir:
- Datos de identificación del paciente.  
- Motivo de consulta (en palabras del paciente, entre comillas y cursiva).  
- Contexto familiar, laboral o social.  
- Primera impresión clínica (afecto, coherencia, actitud, lenguaje, comportamiento).  
- Observaciones iniciales del evaluador. 

2. **Historia del problema (200-300 palabras):**
Incluir:
- Origen, evolución y curso del problema.  
- Factores desencadenantes o agravantes.  
- Antecedentes familiares, médicos y psicológicos relevantes.  
- Factores de riesgo y protectores.  
- Impacto funcional (emocional, social, laboral, físico).  
- Intentos previos de solución o tratamiento.

3. **Evaluación clínica con alternativas diagnósticas (300-500 palabras):**
Objetivo: Proporcionar información suficiente para el análisis clínico, ofreciendo varias alternativas de resultados e interpretaciones plausibles de pruebas, observaciones y entrevistas.

Estructura interna:
a) **Síntomas observados y autoinformados**  
- Describir síntomas emocionales, cognitivos, conductuales y fisiológicos.  
- Presentar **2-3 interpretaciones clínicas plausibles**, en formato fluido.

b) **Resultados de pruebas psicológicas**  
- Simular **3 conjuntos de resultados** (A, B, C) usando instrumentos reales (BAI, BDI-II, PSWQ, STAI, etc.).  
- Mostrar resultados y breves interpretaciones bajo subtítulos claros: *Alternativa A / B / C*.

c) **Observaciones de la entrevista clínica**  
- Presentar **tres interpretaciones posibles** de la observación conductual y actitudinal durante la entrevista.  
- Describir brevemente la conducta del paciente (postura, tono, discurso, afecto) y luego ofrecer lecturas clínicas alternativas coherentes con distintos estilos de afrontamiento o defensas.  
- Las interpretaciones deben diferir en la comprensión del insight o la dinámica emocional.

Ejemplo:  
- *Opción 1:* El paciente se muestra colaborador, con discurso coherente y adecuado insight.  
- *Opción 2:* Minimiza sus síntomas, mantiene rigidez corporal y pobre introspección.  
- *Opción 3:* Usa el humor y la racionalización como mecanismos de defensa ante la ansiedad.

d) **Síntesis descriptiva preliminar**  
- Redactar un párrafo integrador que relacione historia, síntomas y hallazgos.  
- Mantener una hipótesis abierta, sin diagnóstico cerrado.

---

## Criterios de calidad
- Coherencia interna entre historia, síntomas y resultados.  
- Alternativas diagnósticas **plausibles y bien diferenciadas**.  
- Presentación visual clara, técnica y profesional.

---

## Reglas finales
- No incluir diagnóstico definitivo ni plan de intervención.  
- No emitir juicios morales ni interpretaciones personales.  
- Priorizar claridad, rigor técnico y realismo clínico.
"""