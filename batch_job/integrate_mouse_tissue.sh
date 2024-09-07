python /home/jialiulab/disk1/yichun/FuseMap/main.py \
    --input_data_folder_path '/home/jialiulab/disk1/yichun/FuseMap/input_data/mouse_tissue_type_integrate_test1/' \
    --output_save_dir '/home/jialiulab/disk1/yichun/FuseMap/output/mouse_tissue_integrate_test1_llmcombine/' \
    --mode "integrate" \
    --keep_celltype "gtTaxonomyRank4" \
    --keep_tissueregion "gtTissueRegion" \
    --use_llm_gene_embedding "combine"


python /home/jialiulab/disk1/yichun/FuseMap/main.py \
    --input_data_folder_path '/home/jialiulab/disk1/yichun/FuseMap/input_data/mouse_tissue_type_integrate_test1/' \
    --output_save_dir '/home/jialiulab/disk1/yichun/FuseMap/output/mouse_tissue_integrate_test1_llmtrue/' \
    --mode "integrate" \
    --keep_celltype "gtTaxonomyRank4" \
    --keep_tissueregion "gtTissueRegion" \
    --use_llm_gene_embedding "true"


python /home/jialiulab/disk1/yichun/FuseMap/main.py \
    --input_data_folder_path '/home/jialiulab/disk1/yichun/FuseMap/input_data/mouse_tissue_type_integrate_test1/' \
    --output_save_dir '/home/jialiulab/disk1/yichun/FuseMap/output/mouse_tissue_integrate_test1_llmfalse/' \
    --mode "integrate" \
    --keep_celltype "gtTaxonomyRank4" \
    --keep_tissueregion "gtTissueRegion" \
    --use_llm_gene_embedding "false"