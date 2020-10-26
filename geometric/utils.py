import os
import pickle
import numpy as np
import torch_geometric
import torch
from semantic.imports import SceneGraph, SceneGraphObject

'''
Create the feature dictionaries for vertices and edges
'''
def create_embedding_dictionaries(base_dir):

    #Load all Scene Graphs
    sub_dirs=[ dir for dir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,dir))]
    all_scene_graphs=[]
    for sub_dir in sub_dirs:
        all_scene_graphs.extend( pickle.load(open(os.path.join(base_dir,sub_dir,'scene_graphs.pkl'), 'rb')).values() )

    #Gather all vertex and edge types
    vertex_classes=[]
    edge_classes=[]
    for scene_graph in all_scene_graphs:
        for rel in scene_graph.relationships:
            vertex_classes.extend(( rel[0].label, rel[0].color, rel[0].corner ))
            vertex_classes.extend(( rel[2].label, rel[2].color, rel[2].corner ))
            edge_classes.append( rel[1] )
    vertex_classes.append('empty')
    edge_classes.append('attribute')
    
    vertex_classes=np.unique(vertex_classes)
    edge_classes=np.unique(edge_classes)
    
    vertex_embedding_dict={}
    edge_embedding_dict={}

    for i,v in enumerate(vertex_classes):
        #vec=np.zeros(len(vertex_classes), dtype=np.float32)
        #vec[i]=1.0
        vertex_embedding_dict[v]=i

    for i,e in enumerate(edge_classes):
        vec=np.zeros(len(edge_classes), dtype=np.float32)
        vec[i]=1.0   
        edge_embedding_dict[e]=i     

    return vertex_embedding_dict, edge_embedding_dict

def create_scenegraph_data(scene_graph, node_dict, edge_dict):
    node_features=[]
    edges=[]
    edge_features=[]

    #Encode empty SG as: Empty ---attribute---> Empty
    if scene_graph.is_empty():
        node_features.append(node_dict['empty'])
        node_features.append(node_dict['empty'])
        edges.append((0,1))
        edge_features.append(edge_dict['attribute'])
        edges=np.array(edges).reshape((2,1))
        return torch_geometric.data.Data(x=torch.from_numpy(np.array(node_features,dtype=np.int64)),
                                        edge_index=torch.from_numpy(np.array(edges,dtype=np.int64)),
                                        edge_attr= torch.from_numpy(np.array(edge_features,dtype=np.float32)))

    for rel in scene_graph.relationships:
        sub, rel_type, obj=rel

        #Node for the subject
        node_features.append(node_dict[sub.label])
        sub_idx=len(node_features)-1

        #Node for the object
        node_features.append(node_dict[obj.label])
        obj_idx=len(node_features)-1

        #The relationship edge
        edges.append( (sub_idx,obj_idx) ) 
        edge_features.append( edge_dict[rel_type] )

        # Color and Corner for subject
        node_features.append(node_dict[sub.color])
        edges.append( (len(node_features)-1, sub_idx) )
        edge_features.append(edge_dict['attribute'])
        node_features.append(node_dict[sub.corner])
        edges.append( (len(node_features)-1, sub_idx) )
        edge_features.append(edge_dict['attribute'])

        # Color and Corner for object
        node_features.append(node_dict[obj.color])
        edges.append( (len(node_features)-1, obj_idx) )
        edge_features.append(edge_dict['attribute'])
        node_features.append(node_dict[obj.corner])
        edges.append( (len(node_features)-1, obj_idx) )
        edge_features.append(edge_dict['attribute'])        

    node_features=np.array(node_features)
    edges=np.array(edges).T #Transpose for PyG-format
    edge_features=np.array(edge_features)
    assert len(edge_features)==edges.shape[1]== len(scene_graph.relationships) * (2*2+1)
    assert len(node_features)==len(scene_graph.relationships) * (2*3)

    #return np.array(node_features), np.array(edges), np.array(edge_features)
    return torch_geometric.data.Data(x=torch.from_numpy(node_features.astype(np.int64)),
                                     edge_index=torch.from_numpy(edges.astype(np.int64)),
                                     edge_attr= torch.from_numpy(edge_features.astype(np.float32)))

def create_scenegraph_data_co_reference(scene_graph, node_dict, edge_dict):
    node_features=[]
    edges=[]
    edge_features=[]

    #Encode empty SG as: Empty ---attribute---> Empty
    if scene_graph.is_empty():
        node_features.append(node_dict['empty'])
        node_features.append(node_dict['empty'])
        edges.append((0,1))
        edge_features.append(edge_dict['attribute'])
        edges=np.array(edges).reshape((2,1))
        return torch_geometric.data.Data(x=torch.from_numpy(np.array(node_features,dtype=np.int64)),
                                        edge_index=torch.from_numpy(np.array(edges,dtype=np.int64)),
                                        edge_attr= torch.from_numpy(np.array(edge_features,dtype=np.float32)))

    node_objects={} # {SG-Object: node-index}
    num_coref=0

    for rel in scene_graph.relationships:
        sub, rel_type, obj=rel

        #Node for the subject
        new_sub_node=True
        for n in node_objects.keys():
            if np.all(n.center_hash == sub.center_hash):
                sub_idx=node_objects[n]
                new_sub_node=False
                num_coref+=1
                break

        if new_sub_node:
            node_features.append(node_dict[sub.label])
            sub_idx=len(node_features)-1
            node_objects[sub]=sub_idx

        #Node for the object
        new_obj_node=True
        for n in node_objects.keys():
            if np.all(n.center_hash == obj.center_hash):
                obj_idx=node_objects[n]
                new_obj_node=False
                num_coref+=1
                break        

        if new_obj_node:
            node_features.append(node_dict[obj.label])
            obj_idx=len(node_features)-1
            node_objects[obj]=obj_idx

        #The relationship edge
        edges.append( (sub_idx,obj_idx) ) 
        edge_features.append( edge_dict[rel_type] )

        # Color and Corner for subject
        if new_sub_node:
            node_features.append(node_dict[sub.color])
            edges.append( (len(node_features)-1, sub_idx) )
            edge_features.append(edge_dict['attribute'])
            node_features.append(node_dict[sub.corner])
            edges.append( (len(node_features)-1, sub_idx) )
            edge_features.append(edge_dict['attribute'])

        # Color and Corner for object
        if new_obj_node:
            node_features.append(node_dict[obj.color])
            edges.append( (len(node_features)-1, obj_idx) )
            edge_features.append(edge_dict['attribute'])
            node_features.append(node_dict[obj.corner])
            edges.append( (len(node_features)-1, obj_idx) )
            edge_features.append(edge_dict['attribute'])        

    node_features=np.array(node_features)
    edges=np.array(edges).T #Transpose for PyG-format
    edge_features=np.array(edge_features)

    assert len(edge_features)==edges.shape[1]== (2*len(scene_graph.relationships) - num_coref)*2 + len(scene_graph.relationships)
    assert len(node_features)== ( 2*len(scene_graph.relationships) - num_coref) * 3

    #return np.array(node_features), np.array(edges), np.array(edge_features)
    data= torch_geometric.data.Data(x=torch.from_numpy(node_features.astype(np.int64)),
                                     edge_index=torch.from_numpy(edges.astype(np.int64)),
                                     edge_attr= torch.from_numpy(edge_features.astype(np.float32)))    

    return data, num_coref    


def draw_graph(scene_graph):
    pass

if __name__ == "__main__":
    base_dir='data/pointcloud_images_o3d_merged'
    vertex_embedding_dict, edge_embedding_dict=create_embedding_dictionaries(base_dir)
    print(vertex_embedding_dict.keys())
    print(edge_embedding_dict.keys())

    pickle.dump( (vertex_embedding_dict, edge_embedding_dict), open(os.path.join(base_dir, 'graph_embeddings.pkl'),'wb'))

    #scene_graphs=pickle.load(open('data/pointcloud_images_o3d_merged/sg27_station2_intensity_rgb/scene_graphs.pkl','rb'))



