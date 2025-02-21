from zf_pf_diffeo.pipeline import do_HistPointData


if __name__ == "__main__":
    # Define folder paths
    proj_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/projected_surfaces"
    maps_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/Maps"
    
    do_HistPointData(proj_dir,maps_dir,["genotype","condition","time in hpf"],maps_dir,"projected_data","Projected Surface file name")
    