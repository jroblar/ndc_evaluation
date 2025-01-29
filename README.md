# **Plan de Proyecto: Climate Policy Impact Analyzer (CPIA)**  

### **1. Definición del Problema**  
Queremos construir un modelo econométrico/predictivo para evaluar el impacto de las políticas climáticas en la reducción de emisiones de gases de efecto invernadero a nivel país.  

**Objetivos específicos:**  
1. **Construcción del dataset:** Integrar datos de emisiones, políticas climáticas y variables económicas/demográficas de cada país.  
2. **Desarrollo de un índice de rigurosidad en políticas climáticas:** Basado en metodologías como la del Oxford COVID-19 Government Response Tracker.  
3. **Modelado econométrico/predictivo:** Estimar el efecto de las políticas climáticas en la reducción de emisiones.  
4. **Evaluación de umbrales y robustez:** Analizar si existe un nivel de intensidad de políticas necesario para que las emisiones disminuyan significativamente.  
5. **Predicciones:** Determinar la probabilidad de que un país reduzca sus emisiones en un X% en un año determinado.  

---

### **2. Recopilación de Datos**  
| **Fuente** | **Descripción** | **Enlace** |  
|------------|---------------|------------|  
| **Base de emisiones** | Datos históricos de emisiones de cada país | [Descargar](https://1drv.ms/u/s!AqjkGBjI6COCjbw91Mc0ff1Ad4bwjw?e=PpuuWb) |  
| **Base de políticas climáticas** | Políticas ambientales implementadas en cada país | [IEA](https://www.iea.org/policies) |  
| **Datos socioeconómicos** | Variables macroeconómicas y demográficas de cada país | [World Bank](https://data.worldbank.org/) |  

**Tareas:**  
✅ Limpiar y unificar formatos de datos.  
✅ Completar valores faltantes con imputaciones razonables.  
✅ Construir un índice de rigurosidad en políticas climáticas.  

---

### **3. Construcción del Índice de Rigurosidad en Políticas Climáticas**  
**Referencia:** [Oxford COVID-19 Policy Tracker](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md)  

**Pasos:**  
1. **Clasificación de políticas:** Agrupar políticas climáticas en categorías clave (subsidios, impuestos, regulación, incentivos, etc.).  
2. **Ponderación:** Asignar pesos a cada política según su potencial impacto en la reducción de emisiones.  
3. **Normalización:** Crear un índice de 0 a 100, donde 100 representa la máxima rigurosidad en políticas.  
4. **Validación:** Comparar el índice con estudios previos sobre efectividad de políticas climáticas.  

---

### **4. Modelado Econométrico y Predictivo**  
**Modelos candidatos:**  
- **Regresión lineal múltiple**: Para analizar el efecto individual de cada política.  
- **Regresión con variables instrumentales**: Para abordar endogeneidad en la relación política-emisiones.  
- **Modelos de Machine Learning (XGBoost, Random Forest, Deep Learning)**: Para mejorar predicciones.  

✅ Evaluaremos la relación entre políticas y emisiones con regresiones básicas.  
✅ Introduciremos efectos de umbrales con modelos no lineales.  
✅ Implementaremos técnicas de robustez como bootstrap y cross-validation.  

---

### **5. Evaluación de Resultados y Predicciones**  
**Preguntas clave a responder:**  
🔹 ¿Qué políticas han demostrado ser más efectivas?  
🔹 ¿Existe un umbral mínimo de rigurosidad en políticas para ver efectos significativos?  
🔹 ¿Cómo varía la efectividad de las políticas entre países?  
🔹 ¿Qué tan confiables son nuestras predicciones?  

---

### **6. Implementación y Visualización**  
✅ Creación de un dashboard interactivo con visualización de tendencias.  
✅ Predicciones sobre reducción de emisiones en distintos escenarios de políticas.  
✅ Reporte final con hallazgos clave y recomendaciones de política.  

---