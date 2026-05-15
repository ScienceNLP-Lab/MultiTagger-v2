class ParagraphPrompts:
    RULES = (
            "1. Use a formal, objective, scientific tone.\n"
            "2. Never use meta-phrases like 'the authors state' or 'this section describes'.\n"
        )
    CONCISE_GUIDELINES = {
            "background": "Focus on research gap and motivation.",
            "objective": "State the primary aim or hypothesis.",
            "methods": "Detail study design and procedures.",
            "results": "Prioritize key findings and data.",
            "conclusions": "Summarize implications and take-home messages.",
            "none": "Focus on the main point of the paragraph.",
            "intro": "Introduce the main topic and context.",
            "main idea": "Focus on the central concept or hypothesis.",
            "result and conclusion": "Focus on key findings and their implications."   
        }

    DETAILED_GUIDELINES = {
            "background": "Briefly describe the context and significance of the research.",
            "objective": "State the specific aim(s) of the study in a complete sentence.",
            "methods": "Outline the research design, study sample, data collection, and analysis procedures.",
            "results": "Present key findings, including relevant statistics (sample sizes, response rates, P values, confidence intervals). Be specific.",
            "conclusions": "Summarize the main findings and their implications.",
            "none": "Synthesize the core biomedical information, focusing on the primary argument, concept, or supplementary context presented.",
            "intro": "Describe the research context and significance, and clearly state the specific aims or hypotheses.",
            "main idea": "Outline the research design and procedures, while capturing any supplementary methodological context or core concepts.",
            "result and conclusion": "Present key findings with relevant statistics, and summarize their broader implications and take-home messages."   
        }

    @staticmethod
    def bc_prompt(sec_norm, sec_text):
        """
        Basic Concise (BC) prompt.
        """
        
        rules = ParagraphPrompts.RULES

        section_guidelines = ParagraphPrompts.CONCISE_GUIDELINES
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical summarization assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Summarize this {sec_norm if sec_norm != 'none' else 'general'} section\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    @staticmethod
    def di_prompt(sec_norm, sec_text):
        """
        Detailed Instruction Prompt (DI)
        """

        rules = ParagraphPrompts.RULES

        section_guidelines = ParagraphPrompts.DETAILED_GUIDELINES
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical synthesis assistant. "
            f"{rules}"
            "Define your output strictly as:\n"
            "- 'summary': The synthesized biomedical text.\n"
            "- 'reasoning': A brief explanation of how your summary fulfills the given instructions.\n"
            'Format: {"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Synthesize the critical information from this {sec_norm if sec_norm != 'none' else 'general'} section provided in the 'Paragraph text'\n"
            f"{specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    @staticmethod
    def si_prompt(sec_norm, sec_text):
        """
        Structural Instruction Prompt (SI)
        """

        section_guidelines = ParagraphPrompts.CONCISE_GUIDELINES
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "# ROLE\n"
            "You are an expert Biomedical Summarization Assistant.\n\n"
            "# STRICT GUIDELINES\n"
            "- **Tone:** Formal, objective, and academic.\n"
            "- **No Meta-Talk:** Do NOT use phrases like 'The authors state' or 'This section describes'.\n"
            "- **Output Format:** Respond **ONLY** with a valid JSON object. No Markdown blocks, no preamble, no postscript.\n\n"
            "```json\n"
            '{"reasoning": "Brief explanation of how your summary fulfills the given instructions...", "summary": "Final summary..."}\n'
            "```"
        )

        user_msg = (
            f"## TASK: SUMMARY GENERATION\n"
            f"**Target Focus:** {specific_guide}\n\n"
            "**Instructions:** generate a professional biomedical summary.\n"
            f"---\n"
            f"### INPUT TEXT (Reference)\n"
            f"{sec_text}"
        )

        return system_msg, user_msg

    @staticmethod
    def bc_ns_prompt(sec_norm, sec_text):
        """
        Basic Concise prompt for Naive Splitting (BC-NS)
        """
        
        rules = ParagraphPrompts.RULES

        specific_guide = "Summarize the main topic."

        system_msg = (
            "You are a biomedical summarization assistant. "
            f"{rules}"
            "Respond ONLY with a JSON object: "
            '{"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Summarize this section.\n"
            f"Specific focus: {specific_guide}\n\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg
    
    @staticmethod
    def di_trumls_prompt(sec_norm, sec_text, get_umls_terms_textRank, top_umls_terms):
        """
        Detailed Instruction Prompt (DI) with TR-UMLS
        """


        top_entities = get_umls_terms_textRank(sec_text, top_umls_terms)

        rules = ParagraphPrompts.RULES

        section_guidelines = ParagraphPrompts.DETAILED_GUIDELINES
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "You are a biomedical synthesis assistant. "
            f"{rules}"
            "Define your output strictly as:\n"
            "- 'summary': The synthesized biomedical text.\n"
            "- 'reasoning': A brief explanation of how your summary fulfills the given instructions.\n"
            'Format: {"summary": "...", "reasoning": "..."}. '
            "No markdown, no talk."
        )
        
        user_msg = (
            f"Synthesize the critical information from this {sec_norm if sec_norm != 'none' else 'general'} section provided in the 'Paragraph text'\n"
            f"{specific_guide}\n\n"
            f"Ensure the core meanings of these key biomedical entities are preserved or synthesized accurately: {', '.join(top_entities)}\n"
            f"Paragraph text:\n{sec_text}"
        )

        return system_msg, user_msg

    @staticmethod
    def si_cot_prompt(sec_norm, sec_text):
        """
        Structural Instruction Prompt (SI) with CoT
        """

        section_guidelines = ParagraphPrompts.CONCISE_GUIDELINES
        specific_guide = section_guidelines.get(sec_norm.lower(), "Summarize the main topic.")

        system_msg = (
            "# ROLE\n"
            "You are an expert Biomedical Summarization Assistant.\n\n"
            "--- \n"
            "# OPERATIONAL FRAMEWORK\n"
            "You must follow this **2-Stage Chain-of-Thought** process:\n"
            "1. **Stage 1: Element Extraction** (Identify entities, parameters, methodologies, and statistical results).\n"
            "2. **Stage 2: Summary Generation** (Synthesize extracted info into a cohesive, scientific narrative).\n\n"
            "--- \n"
            "# STRICT GUIDELINES\n"
            "- **Tone:** Formal, objective, and academic.\n"
            "- **No Meta-Talk:** Do NOT use phrases like 'The authors state' or 'This section describes'.\n"
            "- **Output Format:** Respond **ONLY** with a valid JSON object. No Markdown blocks, no preamble, no postscript.\n\n"
            "```json\n"
            '{"reasoning": "Stage 1 extraction results...", "summary": "Stage 2 final summary..."}\n'
            "```"
        )

        user_msg1 = (
            f"## TASK 1: ELEMENT EXTRACTION\n"
            f"**Section Type:** `{sec_norm if sec_norm != 'none' else 'general'}`\n"
            "**Instructions:** Extract key elements from the text below, including:\n"
            "- **Entities:** Diseases, genes, drugs, proteins.\n"
            "- **Parameters:** Sample sizes, dosage, duration.\n"
            "- **Methodology:** Study design, assays, equipment.\n"
            "- **Statistics:** P-values, confidence intervals, effect sizes.\n\n"
            f"---\n"
            f"### INPUT TEXT\n"
            f"{sec_text}"
        )
        
        user_msg2 = (
            f"## TASK 2: SUMMARY GENERATION\n"
            f"**Target Focus:** {specific_guide}\n\n"
            "**Instructions:** Using the elements extracted in Task 1, generate a professional biomedical summary.\n"
            "Ensure the summary is dense with information but remains readable and scientifically accurate.\n\n"
            f"---\n"
            f"### INPUT TEXT (Reference)\n"
            f"{sec_text}"
        )

        return system_msg, user_msg1, user_msg2
    
    

class AbstractPrompts:
    @staticmethod
    def refinement_prompt():
        """Refinement prompt for the final abstract."""
        return """You are a biomedical abstract refinement assistant. Refine the abstract based on the abstract draft.
    CRITICAL INSTRUCTION:
    Respond ONLY with a valid JSON object. 
    Do NOT use Markdown code blocks (like ```json). 
    Do NOT provide any conversational text.
    
    Format:
    {
        "abstract": "your abstract text here",
        "reasoning": "your reasoning here"
    }
    """
    