"""Curated local knowledge bundled with Hyperion. No network data is used."""

KNOWLEDGE = [
    {
        "title": "Arquitectura local",
        "tags": ["ia", "local", "privacidad", "arquitectura"],
        "text": "Hyperion es un motor neuro-simbólico local. Combina recuperación semántica mediante embeddings dispersos deterministas, ranking BM25, reglas de intención y un evaluador matemático seguro. Todo se ejecuta en el equipo y ninguna entrada se envía a Internet.",
    },
    {
        "title": "Privacidad",
        "tags": ["privacidad", "datos", "seguridad", "air-gapped"],
        "text": "En modo aislado los mensajes se procesan en memoria local. El historial visible permanece en localStorage del navegador. El servidor no incorpora telemetría, sockets salientes, analítica ni proveedores de IA externos.",
    },
    {
        "title": "Python",
        "tags": ["python", "programación", "código"],
        "text": "Python es un lenguaje interpretado de propósito general. Para código mantenible conviene usar funciones pequeñas, type hints, manejo explícito de excepciones, pruebas automatizadas y entornos reproducibles.",
    },
    {
        "title": "JavaScript",
        "tags": ["javascript", "web", "programación"],
        "text": "JavaScript ejecuta la lógica interactiva del navegador. Las interfaces robustas separan estado, renderizado y acceso a datos; validan entradas y actualizan el DOM sin insertar HTML no confiable.",
    },
    {
        "title": "Complejidad algorítmica",
        "tags": ["algoritmos", "complejidad", "big-o"],
        "text": "La notación Big-O describe cómo escala el coste. O(1) es constante, O(log n) logarítmico, O(n) lineal y O(n²) cuadrático. Debe analizarse tiempo y espacio respecto al tamaño de entrada.",
    },
    {
        "title": "Transformers",
        "tags": ["transformer", "atención", "modelo", "ia"],
        "text": "Un Transformer usa atención para mezclar información entre tokens. La atención de producto punto escalado se expresa como softmax(QKᵀ/√dₖ)V. Entrenar un modelo grande desde cero requiere corpus masivos y hardware especializado.",
    },
    {
        "title": "Embeddings locales",
        "tags": ["embedding", "vectores", "semántica", "búsqueda"],
        "text": "Un embedding representa texto como un vector. Hyperion construye vectores dispersos con tokens y n-gramas, aplica hashing estable y compara similitud coseno. Es una técnica local, ligera y auditable, aunque menos potente que un modelo neuronal grande.",
    },
    {
        "title": "Seguridad de software",
        "tags": ["seguridad", "owasp", "código", "validación"],
        "text": "El software seguro valida entradas, limita recursos, evita ejecutar texto como código, escapa salidas según el contexto, aplica mínimo privilegio y registra errores sin exponer secretos. Las consultas parametrizadas previenen inyección SQL.",
    },
    {
        "title": "Método científico",
        "tags": ["ciencia", "hipótesis", "datos"],
        "text": "El método científico formula una hipótesis falsable, diseña controles, recopila datos, cuantifica incertidumbre y contrasta resultados. Correlación no implica causalidad y una muestra sesgada limita la generalización.",
    },
    {
        "title": "Redes neuronales",
        "tags": ["red neuronal", "machine learning", "entrenamiento"],
        "text": "Una red neuronal ajusta parámetros para minimizar una función de pérdida. El entrenamiento aplica propagación hacia delante, cálculo de gradientes por backpropagation y actualización con un optimizador. Inferencia es usar parámetros ya ajustados.",
    },
    {
        "title": "Git",
        "tags": ["git", "desarrollo", "versiones"],
        "text": "Git registra cambios como commits. Un flujo seguro revisa git diff, ejecuta pruebas, crea commits cohesivos y usa ramas para aislar trabajo. Nunca deben almacenarse credenciales en el repositorio.",
    },
    {
        "title": "APIs",
        "tags": ["api", "http", "servicios"],
        "text": "Una API define un contrato entre componentes. Esta aplicación expone únicamente endpoints HTTP locales para comunicar su interfaz con su propio motor; no consume APIs de IA ni servicios remotos.",
    },
]
