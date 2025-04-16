import os
import torch
import pickle
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops
from neo4j import GraphDatabase

from data.generate_input_from_KG import load_edge, load_node
from data.feature_encoders import IdentityEncoder
from configs.config import Config  # centralized config


class HeteroDataBuilder:
    def __init__(self, config):
        self.config = config
        self.driver = GraphDatabase.driver(config.uri, auth=config.auth)
        self.driver.verify_connectivity()
        os.makedirs(config.output_dir, exist_ok=True)

    def _save_pickle(self, obj, name):
        path = os.path.join(self.config.output_dir, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def build(self):
        data = HeteroData()

        # load cell type nodes from KG
        ct_query = """
        MATCH (ct:CellType)
        RETURN ct.id AS cell_type_name_species, ct.cell_type_name AS cell_type_name
        """
        ct_x, ct_y, ct_mapping, y_mapping = load_node(
            ct_query, index_col="cell_type_name_species", category_col="cell_type_name"
        )

        # ct_mapping is cell type name to index mapping
        # cy_y mapping is cell index yto label mapping
        self._save_pickle(ct_mapping, "ct_mapping.pkl")
        self._save_pickle(y_mapping, "ct_y_mapping.pkl")

        # Load gene nodes from KG
        gene_query = """
        MATCH (gene:Gene)
        RETURN gene.id as gene_id
        """
        gene_x, gene_mapping = load_node(gene_query, index_col="gene_id")
        self._save_pickle(gene_mapping, "gene_mapping.pkl")

        # Load gene is wilcox marker of cell type edges from KG
        marker_query = f"""
        MATCH (g:Gene)-[r:GeneWilcoxMarkerInCellType]->(ct:CellType)
        WHERE r.avg_log2fc >= {self.config.edge_threshold}
        RETURN g.id as gene_id, ct.id as cell_type_name_species, r.avg_log2fc as avg_log2fc
        """
        edge_index, edge_weights = load_edge(
            marker_query,
            src_index_col="gene_id",
            src_mapping=gene_mapping,
            dst_index_col="cell_type_name_species",
            dst_mapping=ct_mapping,
            encoders={"avg_log2fc": IdentityEncoder(dtype=torch.float32)},
        )
        # edge_weights are pytorch tensors for avg_log2FC of marker genes to cell type

        # reverse edge for cell type has marker gene
        # required for GraphConv

        edge_index_rev, edge_weights_rev = load_edge(
            marker_query,
            src_index_col="cell_type_name_species",
            src_mapping=ct_mapping,
            dst_index_col="gene_id",
            dst_mapping=gene_mapping,
            encoders={"avg_log2fc": IdentityEncoder(dtype=torch.float32)},
        )

        # build heterodata
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data["cell_type"].x = torch.eye(len(ct_mapping), device=device)
        data["cell_type"].y = ct_y
        data["gene"].x = torch.eye(len(gene_mapping), device=device)

        data["gene", "is_wilcox_marker_of", "cell_type"].edge_index = edge_index
        data["gene", "is_wilcox_marker_of", "cell_type"].edge_weights = edge_weights

        data["cell_type", "rev_is_wilcox_marker_of", "gene"].edge_index = edge_index_rev
        data["cell_type", "rev_is_wilcox_marker_of", "gene"].edge_weights = (
            edge_weights_rev
        )

        # add self-loops
        if self.config.add_self_loops:
            for node_type in ["cell_type", "gene"]:
                num_nodes = data[node_type].x.size(0)
                loop_index = (
                    torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
                )
                data[node_type, "self_loop", node_type].edge_index = loop_index

        return data

    def save(self, data: HeteroData, filename="mtg_all_sp_wilcox_heterodata.pt"):
        path = os.path.join(self.config.output_dir, filename)
        torch.save(data, path)


if __name__ == "__main__":
    config = Config.data
    builder = HeteroDataBuilder(config)
    hetero_data = builder.build()
    builder.save(hetero_data)
