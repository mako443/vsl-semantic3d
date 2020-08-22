#Calculate the 'inverse' of get_patches_relationship2
#CARE: hope this doesn't cause problems...
def score_triplet(sub,rel_type,obj):
    #CARE: Make sure these match!
    dleft= obj.bbox[0]-(sub.bbox[0]+sub.bbox[2])
    dright= sub.bbox[0]-(obj.bbox[0]+obj.bbox[2])
    dbelow= sub.bbox[1]-(obj.bbox[1]+obj.bbox[3])
    dabove= obj.bbox[1]-(sub.bbox[1]+sub.bbox[3])
    dinfront= (obj.depth-sub.depth)*DEPTH_DIST_FACTOR
    dbehind= (sub.depth-obj.depth)*DEPTH_DIST_FACTOR
    distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)
    score= distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances)
    return np.clip(score,0,1)

#Returns the score and the relationships with object-references instead of label-texts
def ground_scenegraph_to_patches(relations, patches):
    MIN_SCORE=0.1 #OPTION: hardest penalty for relationship not found
    best_groundings=[None for i in range(len(relations))]
    best_scores=[MIN_SCORE for i in range(len(relations))] 

    for i_relation,relation in enumerate(relations): #Walk through relations
        subject_label, rel_type, object_label = relation.sub_label, relation.rel_type, relation.obj_label
        #Walk through all possible groundings
        for subj in [obj for obj in patches if obj.label==subject_label]: 
            for obj in [obj for obj in patches if obj.label==object_label]:
                if subj==obj: continue
                score=score_triplet(subj,rel_type,obj)
                if score>best_scores[i_relation]:
                    best_groundings[i_relation]= Relationship2(subj, rel_type, obj) #(subj,rel_type,obj)
                    best_scores[i_relation]=score

    return np.prod(best_scores), best_groundings    