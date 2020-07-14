Search.setIndex({docnames:["cli","example_report","experiment_series","getting_started","index","modules","mutation_types","numpy_network","params","tasks","torch_network","wann_genetic","wann_genetic.environment","wann_genetic.genetic_algorithm","wann_genetic.individual","wann_genetic.individual.numpy","wann_genetic.individual.torch","wann_genetic.postopt","wann_genetic.postopt.vis","wann_genetic.tasks","wann_genetic.tools"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":2,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["cli.rst","example_report.md","experiment_series.rst","getting_started.rst","index.rst","modules.rst","mutation_types.rst","numpy_network.rst","params.rst","tasks.rst","torch_network.rst","wann_genetic.rst","wann_genetic.environment.rst","wann_genetic.genetic_algorithm.rst","wann_genetic.individual.rst","wann_genetic.individual.numpy.rst","wann_genetic.individual.torch.rst","wann_genetic.postopt.rst","wann_genetic.postopt.vis.rst","wann_genetic.tasks.rst","wann_genetic.tools.rst"],objects:{"":{wann_genetic:[11,0,0,"-"]},"wann_genetic.environment":{environment:[12,0,0,"-"],evaluation_util:[12,0,0,"-"],util:[12,0,0,"-"]},"wann_genetic.environment.environment":{Environment:[12,1,1,""],run_experiment:[12,4,1,""]},"wann_genetic.environment.environment.Environment":{default_params:[12,2,1,""],elite_size:[12,3,1,""],env_path:[12,3,1,""],load_gen_metrics:[12,3,1,""],load_hof:[12,3,1,""],load_indiv_measurements:[12,3,1,""],load_pop:[12,3,1,""],open_data:[12,3,1,""],optimize:[12,3,1,""],pool_map:[12,3,1,""],population_metrics:[12,3,1,""],post_optimization:[12,3,1,""],run:[12,3,1,""],sample_weights:[12,3,1,""],seed:[12,3,1,""],setup_optimization:[12,3,1,""],setup_params:[12,3,1,""],setup_pool:[12,3,1,""],store_data:[12,3,1,""],store_gen:[12,3,1,""],store_gen_metrics:[12,3,1,""],store_hof:[12,3,1,""],stored_indiv_measurements:[12,3,1,""],stored_populations:[12,3,1,""]},"wann_genetic.environment.evaluation_util":{evaluate_inds:[12,4,1,""],express_inds:[12,4,1,""],get_objective_values:[12,4,1,""],make_measurements:[12,4,1,""],update_hall_of_fame:[12,4,1,""]},"wann_genetic.environment.util":{TimeStore:[12,1,1,""],derive_path:[12,4,1,""],env_path:[12,4,1,""],gen_key:[12,4,1,""],get_version:[12,4,1,""],ind_from_hdf:[12,4,1,""],ind_key:[12,4,1,""],load_gen:[12,4,1,""],load_gen_metrics:[12,4,1,""],load_hof:[12,4,1,""],load_ind:[12,4,1,""],load_indiv_measurements:[12,4,1,""],load_pop:[12,4,1,""],make_index:[12,4,1,""],open_data:[12,4,1,""],setup_params:[12,4,1,""],store_gen:[12,4,1,""],store_gen_metrics:[12,4,1,""],store_hof:[12,4,1,""],store_ind:[12,4,1,""],store_pop:[12,4,1,""],stored_generations:[12,4,1,""],stored_indiv_measurements:[12,4,1,""],stored_populations:[12,4,1,""]},"wann_genetic.environment.util.TimeStore":{dt:[12,2,1,""],start:[12,3,1,""],stop:[12,3,1,""],t0:[12,2,1,""],total:[12,2,1,""]},"wann_genetic.genetic_algorithm":{GeneticAlgorithm:[13,1,1,""],InnovationRecord:[13,1,1,""],genetic_operations:[13,0,0,"-"],ranking:[13,0,0,"-"]},"wann_genetic.genetic_algorithm.GeneticAlgorithm":{ask:[13,3,1,""],create_initial_pop:[13,3,1,""],evolve_population:[13,3,1,""],hall_of_fame:[13,2,1,""],mutate:[13,3,1,""],population:[13,2,1,""],rank_population:[13,3,1,""],tell:[13,3,1,""]},"wann_genetic.genetic_algorithm.InnovationRecord":{edge_exists:[13,3,1,""],empty:[13,3,1,""],next_edge_id:[13,3,1,""],next_ind_id:[13,3,1,""],next_node_id:[13,3,1,""]},"wann_genetic.genetic_algorithm.genetic_operations":{add_edge:[13,4,1,""],add_edge_layer_agnostic:[13,4,1,""],add_edge_layer_based:[13,4,1,""],add_node:[13,4,1,""],add_recurrent_edge:[13,4,1,""],add_recurrent_edge_any:[13,4,1,""],add_recurrent_edge_loops_only:[13,4,1,""],change_activation:[13,4,1,""],change_edge_sign:[13,4,1,""],mutation:[13,4,1,""],new_edge:[13,4,1,""],reenable_edge:[13,4,1,""]},"wann_genetic.genetic_algorithm.ranking":{crowding_distances:[13,4,1,""],dominates:[13,4,1,""],rank_individuals:[13,4,1,""]},"wann_genetic.individual":{genes:[14,0,0,"-"],individual_base:[14,0,0,"-"],network_base:[14,0,0,"-"],numpy:[15,0,0,"-"],torch:[16,0,0,"-"],util:[14,0,0,"-"]},"wann_genetic.individual.genes":{Genes:[14,1,1,""],RecurrentGenes:[14,1,1,""]},"wann_genetic.individual.genes.Genes":{copy:[14,3,1,""],edge_encoding:[14,2,1,""],edges:[14,2,1,""],empty_initial:[14,3,1,""],full_initial:[14,3,1,""],n_in:[14,2,1,""],n_out:[14,2,1,""],n_static:[14,3,1,""],node_encoding:[14,2,1,""],node_out_factory:[14,3,1,""],nodes:[14,2,1,""]},"wann_genetic.individual.genes.RecurrentGenes":{edge_encoding:[14,2,1,""]},"wann_genetic.individual.individual_base":{IndividualBase:[14,1,1,""],RecurrentIndividualBase:[14,1,1,""]},"wann_genetic.individual.individual_base.IndividualBase":{Genotype:[14,2,1,""],empty_initial:[14,3,1,""],express:[14,3,1,""],full_initial:[14,3,1,""],get_data:[14,3,1,""],get_measurements:[14,3,1,""],metadata:[14,3,1,""],record_measurements:[14,3,1,""]},"wann_genetic.individual.individual_base.RecurrentIndividualBase":{Genotype:[14,2,1,""]},"wann_genetic.individual.network_base":{BaseFFNN:[14,1,1,""],BaseRNN:[14,1,1,""],NetworkCyclicException:[14,5,1,""],build_weight_matrix:[14,4,1,""],remap_node_ids:[14,4,1,""]},"wann_genetic.individual.network_base.BaseFFNN":{from_genes:[14,3,1,""],index_to_gene_id:[14,3,1,""],is_recurrent:[14,2,1,""],layers:[14,3,1,""],n_act_funcs:[14,3,1,""],n_hidden:[14,3,1,""],n_layers:[14,3,1,""],n_nodes:[14,3,1,""],node_layers:[14,3,1,""],offset:[14,3,1,""],sort_hidden_nodes:[14,3,1,""]},"wann_genetic.individual.network_base.BaseRNN":{from_genes:[14,3,1,""],is_recurrent:[14,2,1,""]},"wann_genetic.individual.numpy":{Individual:[15,1,1,""],RecurrentIndividual:[15,1,1,""],ffnn:[15,0,0,"-"],rnn:[15,0,0,"-"]},"wann_genetic.individual.numpy.Individual":{Phenotype:[15,2,1,""]},"wann_genetic.individual.numpy.RecurrentIndividual":{Phenotype:[15,2,1,""]},"wann_genetic.individual.numpy.ffnn":{Network:[15,1,1,""],apply_act_function:[15,4,1,""],softmax:[15,4,1,""]},"wann_genetic.individual.numpy.ffnn.Network":{activation_functions:[15,3,1,""],available_act_functions:[15,2,1,""],calc_act:[15,3,1,""],get_measurements:[15,3,1,""],measurements_from_output:[15,3,1,""]},"wann_genetic.individual.numpy.rnn":{Network:[15,1,1,""]},"wann_genetic.individual.numpy.rnn.Network":{get_measurements:[15,3,1,""]},"wann_genetic.individual.torch":{Individual:[16,1,1,""],RecurrentIndividual:[16,1,1,""],ffnn:[16,0,0,"-"],rnn:[16,0,0,"-"]},"wann_genetic.individual.torch.Individual":{Phenotype:[16,2,1,""]},"wann_genetic.individual.torch.RecurrentIndividual":{Phenotype:[16,2,1,""]},"wann_genetic.individual.torch.ffnn":{ConcatLayer:[16,1,1,""],MultiActivationModule:[16,1,1,""],Network:[16,1,1,""]},"wann_genetic.individual.torch.ffnn.ConcatLayer":{forward:[16,3,1,""]},"wann_genetic.individual.torch.ffnn.MultiActivationModule":{forward:[16,3,1,""]},"wann_genetic.individual.torch.ffnn.Network":{available_act_functions:[16,2,1,""],get_measurements:[16,3,1,""],measurements_from_output:[16,3,1,""]},"wann_genetic.individual.torch.rnn":{Network:[16,1,1,""],ReConcatLayer:[16,1,1,""]},"wann_genetic.individual.torch.rnn.Network":{get_measurements:[16,3,1,""]},"wann_genetic.individual.torch.rnn.ReConcatLayer":{forward:[16,3,1,""]},"wann_genetic.individual.util":{num_used_activation_functions:[14,4,1,""],rearrange_matrix:[14,4,1,""]},"wann_genetic.postopt":{report:[17,0,0,"-"],vis:[18,0,0,"-"]},"wann_genetic.postopt.report":{Report:[17,1,1,""],compile_report:[17,4,1,""],draw_network:[17,4,1,""],plot_gen_lines:[17,4,1,""],plot_gen_quartiles:[17,4,1,""]},"wann_genetic.postopt.report.Report":{add:[17,3,1,""],add_fig:[17,3,1,""],add_gen_line_plot:[17,3,1,""],add_gen_metrics:[17,3,1,""],add_gen_quartiles_plot:[17,3,1,""],add_image:[17,3,1,""],add_ind_info:[17,3,1,""],add_network:[17,3,1,""],compile:[17,3,1,""],compile_stats:[17,3,1,""],gen_metrics:[17,3,1,""],rel_path:[17,3,1,""],run_evaluations:[17,3,1,""],write_main_doc:[17,3,1,""],write_stats:[17,3,1,""]},"wann_genetic.postopt.vis":{vis_network:[18,0,0,"-"]},"wann_genetic.postopt.vis.vis_network":{draw_graph:[18,4,1,""],draw_weight_matrix:[18,4,1,""],node_names:[18,4,1,""]},"wann_genetic.tasks":{base:[19,0,0,"-"],image:[19,0,0,"-"],load_iris:[19,4,1,""],rnn:[19,0,0,"-"],select_task:[19,4,1,""]},"wann_genetic.tasks.base":{ClassificationTask:[19,1,1,""],RecurrentTask:[19,1,1,""],Task:[19,1,1,""]},"wann_genetic.tasks.base.ClassificationTask":{get_data:[19,3,1,""],load:[19,3,1,""],test_x:[19,2,1,""],test_y:[19,2,1,""],x:[19,2,1,""],y:[19,2,1,""],y_labels:[19,3,1,""]},"wann_genetic.tasks.base.RecurrentTask":{is_recurrent:[19,2,1,""]},"wann_genetic.tasks.base.Task":{get_data:[19,3,1,""],is_recurrent:[19,2,1,""],load:[19,3,1,""],load_test:[19,3,1,""],load_training:[19,3,1,""]},"wann_genetic.tasks.image":{deskew:[19,4,1,""],digit_raw:[19,4,1,""],mnist_256:[19,4,1,""],preprocess:[19,4,1,""]},"wann_genetic.tasks.rnn":{AddingTask:[19,1,1,""],CopyTask:[19,1,1,""],EchoTask:[19,1,1,""]},"wann_genetic.tasks.rnn.AddingTask":{get_data:[19,3,1,""],load:[19,3,1,""],n_in:[19,2,1,""],n_out:[19,2,1,""],sample_length:[19,2,1,""]},"wann_genetic.tasks.rnn.CopyTask":{T:[19,2,1,""],get_data:[19,3,1,""],load:[19,3,1,""],num_categories:[19,2,1,""],rep_seq_len:[19,2,1,""]},"wann_genetic.tasks.rnn.EchoTask":{get_data:[19,3,1,""],load:[19,3,1,""]},"wann_genetic.tools":{cli:[20,0,0,"-"],compare_experiments:[20,0,0,"-"],experiment_series:[20,0,0,"-"]},"wann_genetic.tools.cli":{generate_experiments:[20,4,1,""]},"wann_genetic.tools.compare_experiments":{load_series_stats:[20,4,1,""],mean_comparison:[20,4,1,""]},"wann_genetic.tools.experiment_series":{ExperimentSeries:[20,1,1,""],Variable:[20,1,1,""]},"wann_genetic.tools.experiment_series.ExperimentSeries":{assemble_stats:[20,3,1,""],configuration_env:[20,3,1,""],configuration_name:[20,3,1,""],configuration_params:[20,3,1,""],configurations:[20,3,1,""],create_experiment_files:[20,3,1,""],discover_data_dir:[20,3,1,""],experiment_paths:[20,2,1,""],flat_values:[20,3,1,""],from_spec_file:[20,3,1,""],init_variables:[20,3,1,""],num_configurations:[20,3,1,""],var_names:[20,3,1,""],vars:[20,3,1,""]},"wann_genetic.tools.experiment_series.Variable":{iter_indices:[20,3,1,""],set_value:[20,3,1,""],value:[20,3,1,""],value_name:[20,3,1,""]},"wann_genetic.util":{ParamTree:[11,1,1,""],get_array_field:[11,4,1,""],nested_update:[11,4,1,""]},"wann_genetic.util.ParamTree":{params:[11,3,1,""],update_params:[11,3,1,""],update_params_at:[11,3,1,""]},wann_genetic:{environment:[12,0,0,"-"],genetic_algorithm:[13,0,0,"-"],individual:[14,0,0,"-"],postopt:[17,0,0,"-"],tasks:[19,0,0,"-"],tools:[20,0,0,"-"],util:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function","5":"py:exception"},terms:{"06464v4":9,"16x16":19,"28x28":19,"6976018327eee":9,"8x8":19,"9ea7ceaaa3765f536d95":12,"boolean":19,"case":[2,8],"class":[5,11,12,13,14,15,16,17,19,20],"default":[0,3,5,8,11,12,14],"final":[2,7,10],"function":[0,2,6,7,8,10,13,14,15,16],"import":2,"int":[12,14,20],"long":[5,6,9,14],"new":[4,12,13,14],"return":[11,12,13,14,15,19,20],"true":[3,5,8,12,14,19,20],"var":20,"while":[2,10,16],Adding:19,For:[3,5,7,14,15],One:[2,3],The:[2,3,6,7,8,9,10,12,13,14],There:6,Use:[2,12],Used:0,Using:8,Will:14,_fmt:2,_run:[],abc:[5,11,20],abs:[9,15,16],abspath:17,access:11,accord:[13,14],accuraci:[1,8,12,20],across:6,act:2,activ:[0,3,4,7,8,10,13,14,15,16],activation_funct:15,active_nod:15,actual:6,acycl:7,add:[0,3,4,8,13,17],add_edg:13,add_edge_layer_agnost:13,add_edge_layer_bas:13,add_fig:17,add_gen_line_plot:17,add_gen_metr:17,add_gen_quartiles_plot:17,add_imag:17,add_ind_info:17,add_network:17,add_nod:13,add_recurrent_edg:13,add_recurrent_edge_ani:13,add_recurrent_edge_loops_onli:13,add_to_sum:15,added:6,adding:2,addingtask:19,addit:2,addition:[5,7,14],additionali:7,affect:6,after:8,afterward:16,age:12,agnost:6,algorithm:[6,7,8,13],alia:[14,15,16],all:[2,3,6,7,8,12,13,14,16,20],all_act_func:16,allow:11,alphabet:9,alreadi:6,also:[2,6,7],altern:8,although:16,alwai:14,amar:9,analysi:4,ani:[2,6,8,12,14,20],anlys:20,anoth:2,api:4,appli:[2,7,13,15,16],apply_act_funct:15,approach:6,arg:[14,15,16],argument:2,arjovski:9,arrai:[11,13,14,19],arrang:6,arxiv:9,as_list:14,asb15:9,ask:13,assemble_stat:20,assum:13,avail:[3,8,9,12,20],available_act_funct:[15,16],available_func:[14,15],axi:15,back:8,backend:[8,12],base:[2,5,6,8,9,11,12,13,14,15,16,17,20],base_param:[2,5,20],base_weight:15,baseffnn:[5,14,15,16],basernn:[14,15,16],basic:[10,19],been:[3,5,6,8,12,14],begin:10,bengio:9,best:[0,3],better:13,between:2,bia:[7,10,14],bigger:10,bin:3,binari:3,birth:[1,5,14,15,16],blob:19,block:10,bool:[5,12,13,14],bound:8,brain:[13,19],build:[0,2,10,14,20],build_dir:[0,2],build_weight_matrix:14,built:16,calc_act:15,calcul:[6,7,10,12,13],call:[2,16],calucl:12,can:[2,3,5,6,7,8,10,13,14,15,20],caption:17,care:[5,12,16],categori:9,certain:2,chang:[2,4,5,8,13,14],change_activ:[12,13],change_edge_sign:[12,13],check:[6,20],classif:4,classificationtask:19,classify_gym:19,classmethod:[5,13,14,20],cli:[0,5,11],collect:[3,5,11,14,20],column:[2,7],com:[3,11,12,13,19],combin:[2,8],command:[2,3,4,12,20],comment:0,commit_elite_freq:[8,12],commit_metrics_freq:[8,12],compar:2,compare_experi:[5,11],comparison:3,compil:[0,3,8,17],compile_report:[0,3,8,12,17],compile_stat:17,complet:[2,3,12],comput:[5,6,14,15,16],concaten:[6,10],concatlay:[10,16],conclus:2,condit:10,config:[4,12],configur:[6,8,20],configuration_env:20,configuration_nam:20,configuration_param:20,conn_mat:[5,14,15],connect:[5,6,7,8,13,14,16],consid:2,consist:[9,14],constitut:12,contain:[2,3,5,6,7,12,14,20],containig:19,contaten:16,content:5,contin:14,control:[2,8],converg:9,convers:10,convert:[3,12,14,19],copi:[2,14,19],copytask:19,corr:9,correspond:[6,7],cos:[15,16],could:[3,6],count:13,creat:[0,2,3,12,13,14,20],create_experiment_fil:20,create_initial_pop:13,crop:19,cross:[2,10,12],crossov:13,crowd:13,crowding_dist:13,culling_ratio:[8,12],current:[0,7,10,12,13],current_gen:14,cyclic:14,dash:7,data:[2,3,5,8,12,20],data_base_path:[8,12],data_path:[2,5,20],datafram:12,dataset:12,deactiv:3,dead:6,debug:[8,12],default_param:12,defin:[2,8,16],delai:19,delimit:9,denot:19,depend:13,depth:11,deriv:[9,12],derive_path:12,describ:[6,9,19],descript:8,deskew:19,deskewd:19,deskw:19,dest:[13,14],detail:[0,8,13],determin:[2,3,7,8],deviat:12,diagram:10,dict:[5,11,12,20],dictionari:[2,5,11,12],differ:[2,5,13,14],digit:19,digit_raw:19,dim:[15,19],dir_path:20,directori:[0,2,3,8,20],disabl:[4,8,13,14],discover_data_dir:20,discuss:13,displai:0,distanc:13,distribut:[2,3,8,12],doc:6,doc_nam:17,document:[3,6,9],doe:[5,6,7,8,14],domain:19,domin:13,don:[7,8],done:[2,3],draw:[0,2],draw_graph:18,draw_network:[0,17],draw_weight_matrix:18,drawn:0,dtype:14,dure:[0,2,3,7,8,9],each:[2,3,8,9,10,13,14,15,20],earlier:[5,14],earliest:[5,6,14],easier:10,easili:2,echo:[2,3,19],echotask:19,edg:[1,2,4,5,8,10,13,14,15],edge_encod:14,edge_exist:13,eeedeeeeeeeee:9,eeee6976018327:9,eeeeeeeeeeeee:9,effici:6,egdg:6,eight:9,either:[6,8],element:[7,9,17],elementwis:16,elit:[6,8,12],elite_ratio:[8,12],elite_s:12,els:[5,11,14],empti:[8,9,13],empty_initi:14,enabl:[6,8,12,13,14],enable_edge_sign:[8,12],enabled_activation_func:[8,12],enabled_activation_funct:[],encod:[5,14],encount:14,end:6,enter:6,entir:2,entri:[0,8],env:[12,13,17,19],env_path:12,environ:[3,5,8,11],equal:12,equival:13,essenti:[5,6,14],evalu:[0,3,5,8,12],evaluate_ind:12,evaluation_util:[5,11],ever:13,everi:[2,8,16],everyth:12,evolut:[9,19],evolutionari:[8,13],evolv:12,evolve_popul:13,exampl:[2,4],except:14,exclud:6,execut:[0,2,4,5,8,12,20],exist:[6,10,11,13,20],exit:[0,2],expect:[2,9,20],experi:[4,5,8,12,13,20],experiment_nam:[2,8],experiment_path:20,experiment_seri:[5,11],experimentseri:[5,20],explain:2,explan:15,explicitli:7,express:[7,12,14],express_ind:12,fall:8,fals:[8,12,13,14,19],fame:[0,3,8,12],faster:2,fed:10,feed:[4,5,14,15,16],few:6,ff_weight:16,ff_weight_mat:[],ffn:15,ffnn:[11,14],field:[0,2,8,11,20],figur:[3,8],file:[0,2,3,5,8,12,20],filenam:8,first:[4,7,14],fit:14,flag:19,flat_valu:20,flatten:20,flow:6,fmt:[2,20],fname:17,follow:[3,8,10],format:[2,8],former:16,forward:[4,5,14,15,16],found:[8,20],frame:20,frequenc:8,frequent:8,from:[0,2,3,5,7,9,10,12,14,20],from_gen:14,from_spec_fil:20,front_object:13,full:[8,12,20],full_initi:14,fullfil:[5,14,15],fulli:[10,14],func:[12,14],function_nam:0,function_plot:0,further:8,gaussian:[15,16],gen:12,gen_data:12,gen_kei:12,gen_metr:17,gene:[5,6,7,10,11,12,13,15,16],gener:[3,4,5,7,8,12,20],generate_experi:20,generate_experiment_seri:[0,2],genet:[5,6,14],genetic_algorithm:[5,11],genetic_oper:[5,11],geneticalgorithm:13,genotyp:14,get:[0,2,4,10,12,14,20],get_array_field:11,get_data:[14,19],get_measur:[14,15,16],get_objective_valu:12,get_vers:12,gist:12,git:3,github:[3,12,13,19],give:8,given:[0,5,14,20],good:13,googl:[13,19],graph:0,green:7,group_valu:20,group_var:20,guid:3,h5py:12,hall:[0,3,8,12],hall_of_fam:[3,13],has:[3,6,7,8,20],have:[5,6,7,8,10,12,14],hdf5:[3,8,12],hdf:12,help:[0,6],helper:19,henc:[5,14],hidden:[1,5,7,13,14],hidden_lay:[5,14,15],hierach:14,hierarch:7,hof_evaluation_iter:[8,12],hof_metr:[8,12],hof_siz:[8,12],hold:[8,10],hook:16,how:[2,3,6,8,12,15],html:3,http:[9,11,12,13,19],human:3,hyper:[2,3,8],ident:[3,15,16],ids:[12,13,14],ids_onli:12,ignor:[8,16],imag:[3,5,11],image_shap:19,img:19,impact:2,implement:[4,5,6,9,13,14,15,16],implic:6,implment:[15,16],includ:[2,3,12,14],include_input:14,include_output:14,include_prediction_record:[],incom:[5,6,14],ind:[12,13,17],ind_from_hdf:12,ind_kei:12,indec:14,index:[7,12,13,14,17,20],index_to_gene_id:14,indic:[8,13,14],indiv_measur:12,individu:[0,3,5,8,10,11,12,13],individual_bas:[5,11,15,16],individualbas:[14,15,16],influenc:8,inform:3,init_vari:20,initi:[7,8,13,14,20],initial_enabled_edge_prob:[8,12],initial_gen:[8,12],innov:13,innovationrecord:13,input:[6,7,8,9,10,14,16,19],insert:3,insid:3,instal:[0,4],instanc:16,instead:[2,10,12,13,16],int64:14,inter:3,interfac:[3,4,12],introduc:13,introduct:13,invers:[15,16],invok:3,iri:[1,2,3,12],is_recurr:[14,19],issu:13,iter:[2,3,8,12,20],iter_indic:20,iterat:20,iteratiion:8,json:[3,12,17],just:[2,8,10,13],kappa:[1,8,12],keep:[2,10,13],kei:[1,2,11,12,14,20],kwarg:[11,14,15,16],label:[0,17,18],lambda:[15,16],larger:6,last:[9,14],lastli:2,later:8,latest:[5,6,14],latest_poss:[5,14],latter:[6,16],layer:[1,5,6,7,10,13,14],layer_agnost:[8,12],layer_bas:8,layer_h:18,lead:[6,14],least:13,length:[],less:6,level:[2,8],like:[2,3,10],limit:10,line:[2,3,4,7,12,20],linear:10,list:[2,8,11,12,13],load:[12,19,20],load_experiment_seri:2,load_func:19,load_gen:12,load_gen_metr:12,load_hof:12,load_ind:12,load_indiv_measur:12,load_iri:19,load_pop:12,load_series_stat:[2,20],load_test:19,load_train:19,locat:2,log:[3,8,12],log_filenam:[8,12],log_loss:[1,8,12],lognorm:[2,3],look:[2,3,7],loop:[2,13],loops_onli:8,lower:[8,10],lower_bound:[2,8,12],machin:2,made:12,maintain:6,make:[2,3,5,6,10,14],make_index:12,make_measur:12,mani:[5,12,14],map:[2,5,11,14,20],mark:7,markdown:3,martin:9,mask:10,master:19,match:20,math:14,matric:7,matrix:[5,7,14],max:[1,12],maxim:[8,12,13],maximum:[6,10,12],mean:[1,2,3,8,12,20],mean_comparison:20,measur:[0,1,3,8,12,13,14,15,16,17,20],measurements_from_output:[15,16],media:3,median:12,memor:9,memori:[5,6,7,14],messag:0,met:10,metadata:[12,14],method:[13,16,19],metric:[0,3,8,12],metric_nam:12,middl:13,might:[2,3,6,10],min:[1,8,12],mind:10,minim:[3,8,12],minimum:[5,6,12,14],mnist:19,mnist_256:19,mode:12,modul:5,moment:[3,5,6,14,19],more:[2,6],much:[2,8],mulitpl:0,multi:10,multiactivationmodul:[15,16],multipl:[0,2,6,8,13,16,20],multipli:7,multivari:20,must:2,mutablemap:11,mutat:[1,4,5,12,13,14,15,16],n_act_func:14,n_hidden:[8,12,14],n_in:[5,14,15,19],n_layer:14,n_node:14,n_out:[5,14,15,19],n_sampl:12,n_static:14,name:[0,2,3,8,12,17,20],ndarrai:[5,11,12,13,14],necessari:2,need:[2,3,7,9,10,13,16],neg:[8,12,16],negat:19,negative_edges_allow:14,nest:11,nested_upd:11,net:18,network:[0,2,4,5,6,9,12,14,15,16,19],network_bas:[5,11,15,16],networkcyclicexcept:14,neural:[2,4,5,9,12,14,15,16,19],new_edg:[12,13],new_nod:12,new_recurrent_edg:12,next:7,next_edge_id:13,next_ind_id:13,next_node_id:13,nice:2,node:[0,1,4,5,7,8,10,13,14,15,16],node_act_func:16,node_encod:14,node_lay:14,node_nam:18,node_out_factori:14,none:[5,11,12,13,14,15,16,17,18,19,20],normal:[8,15],notabl:[5,14],nsga:13,num_categori:[12,19],num_configur:20,num_gener:[3,8,12],num_sampl:[3,8,12,17],num_samples_per_iter:[8,12],num_training_samples_per_iter:[],num_used_activation_funct:14,num_weight:[3,8,12,17],num_weight_samples_per_iter:3,num_weights_per_iter:[8,12],num_work:[8,12],number:[0,1,3,5,8,10,13,14,20],numpi:[4,5,8,10,11,12,13,14,19],obj_valu:13,object:[8,12,13,14,16,17,19,20],occur:8,offset:14,offspr:8,onc:[3,5,6,10,14],one:[2,3,8,10,13,16],onli:[2,5,6,7,8,10,12,13,14],open:12,open_data:12,optim:[6,8,12],option:[0,5,6,8,12,13,14],optiz:12,order:[2,6,7,13,20],org:9,origin:[10,13],other:[2,6,7,8],out:[9,14],outdir:20,outgo:6,output:[0,2,3,6,7,8,9,10,14,16],over:[3,8],overridden:16,overview:[3,4,8],overwritten:2,own:2,packag:[3,4,5],pair:6,panda:[12,20],pandoc:3,parallel:8,param:[0,2,3,5,6,8,11,12,13,14,15,19,20],paramet:[2,3,4,5,6,11,12,13,14,20],params_map:2,paramtre:[11,12,20],parent:[5,14,15,16],pareto:13,part:[2,12,20],pass:[2,16],patchcorn:19,patchdim:19,path:[0,2,3,5,12,17,20],path_to_series_spec:2,pdf:3,perform:[3,11,14,16],phase:9,phenotyp:[14,15,16],pick:6,pip:3,pixel:19,place:[0,2],plonerma:3,plot:[3,4],plot_gen_lin:[0,17],plot_gen_quartil:[0,17],point:0,pool:[5,12],pool_map:12,pop:[8,12],popul:[3,4,12,13],population_metr:12,pos_iter:18,posit:[6,12],possibl:[5,6,8,10,14,20],post:[3,5,8,12],post_init_se:[8,12],post_optim:12,postopt:[3,4,5,11,12],potenti:[5,6,14],predict:[15,16],prefer:[5,6,14],prefix:[2,8],preprocess:19,present:2,previous:[5,20],prior:[7,16],prob_en:14,probabl:[2,8,12],problem:[5,8,14],problemat:10,process:[2,5,7,8,12],produc:[2,3,6,8,12,13],product:[2,12,20],prop_step:[],propag:[7,15],properti:[11,12,14,17,19],provid:[2,3,8,14],prune:6,python3:3,python:2,q_0:12,q_2:12,q_3:12,quartil:[0,12],question:11,rais:14,random:[2,3,6,8],randomli:[6,13,14],rank:[5,8,11],rank_individu:13,rank_popul:13,ratio:[8,12],raw:12,re_weight:16,re_weight_mat:[],reach:7,read:[2,5,12,20],readabl:3,rearrang:14,rearrange_matrix:14,recip:16,recommend:[2,3],reconcatlay:16,record:[8,13],record_measur:14,recorded_metr:[8,12],recurr:[4,8,13,14,16,19],recurrent_weight_matrix:[14,15],recurrentgen:14,recurrentindividu:[15,16],recurrentindividualbas:[14,15,16],recurrenttask:[9,19],recurrr:15,reduc:[5,6,14],reenabl:[4,13],reenable_edg:[12,13],refer:[4,8],referenc:[2,11],regardless:13,regist:[0,16],rel:12,rel_path:17,relev:7,relu:[15,16],remap_node_id:14,remov:[6,10],rep_seq_len:19,replac:14,report:[4,5,8,11,12],report_nam:17,repositori:3,repres:[5,7,9,14,20],represent:14,reproduc:9,requir:[2,3,5,6,8,12,14],resiz:19,restrict:10,result:[2,3,6],retriev:12,return_indiv_measur:12,return_ord:13,rewann:[],rewannmodul:[],rnn:[5,11,14],robbieclarken:12,row:7,rsvg:3,run:[1,2,3,4,12,16],run_evalu:17,run_experi:[0,2,3,12],run_postopt:[3,8,12],same:[0,2,6,7,15],sampel:[3,12],sampl:[0,2,3,4,6,12,19],sample_length:[12,19],sample_order_se:12,sample_weight:12,save:20,schedul:[5,14],scikit:19,score:15,script:20,second:7,section:[2,3,8],see:[0,2,3,6,8,12,13,15,19],seed:[2,3,8,12],select:[4,6,10,12,13,15],select_task:19,selected_func:15,sens:6,sequenc:[9,20],seri:[3,4,5,8,20],series_spec_fil:0,set:[3,6,7,8,12,13,15],set_valu:20,setup:12,setup_optim:12,setup_param:12,setup_pool:12,shah:9,shape:19,share:[2,15],shared_weight:16,should:[8,13,16],show:0,sigma:[2,3,8,9,12],sigmoid:[15,16],sign:[2,4,8,12,13,14],signal:[6,7],silent:16,simpl:[0,2,3],sin:[15,16],sinc:[6,7,16],singl:[0,2],size:[3,6,8,10,12,19],skip:10,slightli:[2,13],softmax:15,some:[3,6,14],someth:2,sort:[5,6,7,10,13,14],sort_hidden_nod:[5,14],sourc:[5,6,11,12,13,14,15,16,17,18,19,20],space:9,span:6,spec:[2,3,5,8,12,20],spec_path:20,specif:[0,3,4,8,20],specifi:[2,3,12],split:13,squar:[15,16],src:[13,14],ssh:3,stackoverflow:11,standard:[12,15,16],start:[0,4,6,8,10,11,12,13],start_id:13,stat:[3,17,20],state:7,statist:[2,3,8,12,14],std:12,step:[7,15,16],stop:12,storag:[4,12],store:[0,3,7,8,11,12],store_data:12,store_gen:12,store_gen_metr:12,store_hof:12,store_ind:12,store_pop:12,stored_gener:12,stored_indiv_measur:12,stored_popul:12,str:[2,5,11,12,14,20],strategi:[8,12,13],string:[2,12],structur:[3,14],subclass:16,subdirectori:[2,3,12],subject:8,submodul:5,subpackag:5,subset:8,suffici:2,sum:[8,15],suppos:[3,9],sure:3,suriv:13,surviv:8,svg:3,symbol:9,synthet:8,system:3,take:[2,3,5,6,10,12,16],tanh:[15,16],target:[2,6],task:[2,3,4,5,6,11,12,13],task_nam:[8,19],tell:[13,19],templat:[0,2],tensor:16,test:[3,8,12,19,20],test_i:19,test_port:12,test_x:19,than:6,thei:[0,2,8],them:[6,16],therefor:7,thi:[0,2,3,5,6,7,8,9,10,12,13,14,15,16,19,20],those:9,through:2,time:6,timestor:12,toi:19,tokyo:[13,19],toml:[2,3,8,20],tool:[0,2,5,11],top:2,topolog:[5,6,7,10,14],torch:[4,5,6,8,11,14,15],torch_network:10,total:[12,14],tournament:[8,13],tournament_s:[8,12],track:[3,13],train:[3,5,8,12],translat:14,travers:10,treat:8,tree:12,trivial:6,tupl:[12,14,19,20],two:[5,6,7,13,14],type:[2,4,8,12,13,14,16],uniform:[2,8,12],unitari:[9,19],unskew:19,updat:[11,12,14,15],update_hall_of_fam:12,update_param:11,update_params_at:11,upper:[8,10],upper_bound:[2,8,12],usag:[0,2],use:[0,2,3,5,8,12,14,20],use_bas:20,use_tourna:[8,12],used:[2,3,5,7,8,10,12,13,14,20],useful:[2,20],userdict:11,using:[0,2,3,8,10,13,19],util:[5,20],valid:8,valu:[1,2,3,7,11,15,20],value_nam:20,var_nam:20,vari:[8,11],variabl:[2,20],variant:8,variat:[13,20],variou:2,vector:[6,7,10,14],venv:3,versa:6,via:[2,3,10,11],vice:6,virtual:3,vis:[10,11,17],vis_network:[11,17],vsvinayak:19,wait:9,wann:[6,15,16,19],wann_genet:[0,2,3,4,5,8,9,10],wannreleas:19,want:[2,8],weigh:2,weight:[0,3,6,7,8,12,14,15,16],weight_matrix:[5,14,15],well:[10,12],went:2,were:[3,12,20],what:[5,14],when:[6,7,8,14],where:13,whether:[8,12,13,19,20],which:[2,3,8,9,12,14],wich:3,wise:3,within:[2,8,16],without:8,won:14,work:[0,8,15],workshop:[13,19],worst:8,would:[2,6,14],wrap:11,write:[8,12],write_main_doc:17,write_stat:17,wrong:2,y_label:19,y_raw:[15,16],y_true:[15,16],yet:[6,12],yield:14,yoshua:9,you:[2,8],zero:7},titles:["Command Line Interfaces","Report iris_run","Generating Series of Experiments","Getting Started","Genetic Search for WANNs","API","Mutation Types","Numpy Implementation of WANNs","Parameters","Tasks","Torch Implementation of WANNs","wann_genetic package","wann_genetic.environment package","wann_genetic.genetic_algorithm package","wann_genetic.individual package","wann_genetic.individual.numpy package","wann_genetic.individual.torch package","wann_genetic.postopt package","wann_genetic.postopt.vis package","wann_genetic.tasks package","wann_genetic.tools package"],titleterms:{"new":6,Adding:9,activ:6,add:6,add_recurrent_edg:8,analysi:2,api:5,base:19,best:1,chang:6,change_activ:8,change_edge_sign:8,classif:9,cli:20,command:0,compare_experi:20,config:8,confus:1,content:[11,12,13,14,15,16,17,18,19,20],copi:9,disabl:6,echo:9,edg:6,environ:12,evaluation_util:12,exampl:3,execut:3,experi:[0,2,3],experiment_seri:20,fame:1,feed:[7,10],ffnn:[15,16],first:3,forward:[7,10],gene:14,gener:[0,2],genet:4,genetic_algorithm:13,genetic_oper:13,get:3,hall:1,imag:19,implement:[7,10],individu:[1,14,15,16],individual_bas:14,instal:3,interfac:0,iri:9,iris_run:1,line:0,matrix:1,mnist:9,modul:[11,12,13,14,15,16,17,18,19,20],mutat:[6,8],name:9,network:[1,7,10],network_bas:14,neural:10,new_edg:8,new_nod:8,node:6,numpi:[7,15],orign:9,overview:5,packag:[11,12,13,14,15,16,17,18,19,20],paramet:8,plot:0,popul:8,postopt:[8,17,18],rank:13,recurr:[6,7,9],reenabl:6,reenable_edg:8,refer:9,report:[0,1,3,17],result:1,rewann:[],rnn:[15,16,19],run:0,sampl:8,search:4,select:8,seri:[0,2],sign:6,specif:2,start:3,storag:8,submodul:[11,12,13,14,15,16,17,18,19,20],subpackag:[11,14,17],task:[8,9,19],tool:20,torch:[10,16],type:6,util:[11,12,14],virtualenv:3,vis:18,vis_network:18,wann:[4,7,10],wann_genet:[11,12,13,14,15,16,17,18,19,20]}})