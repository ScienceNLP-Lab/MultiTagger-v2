class GetUMLS:
    
    def __init__(self, nlp, linker):
        self.nlp = nlp
        self.linker = linker

    def get_umls(self, text, top_n=5):
        doc = self.nlp(text)
        term_map = {} # Key: Canonical Name, Value: {original_texts: set(), rank: float}

        for phrase in doc._.phrases:
            for ent in phrase.chunks:
                if ent._.kb_ents:
                    cui = ent._.kb_ents[0][0]
                    canon_name = self.linker.kb.cui_to_entity[cui].canonical_name
                    
                    if canon_name not in term_map:
                        term_map[canon_name] = {"original": set(), "rank": phrase.rank}
                    
                    term_map[canon_name]["original"].add(ent.text)
                    term_map[canon_name]["rank"] = max(term_map[canon_name]["rank"], phrase.rank)

        sorted_items = sorted(term_map.items(), key=lambda x: x[1]['rank'], reverse=True)[:top_n]
        
        formatted_terms = []
        for _, info in sorted_items:
            orig = list(info['original'])[0] 
            formatted_terms.append(orig)
        
        return formatted_terms
    