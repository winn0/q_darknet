Quantize=['Quantize']
AdaptiveAvgPool2d=['AdaptiveAvgPool2d']
QuantizedLinear=['QuantizedLinear']
QuantizedConvReLU2d=['QuantizedConvReLU2d']
QuantizedConv2d=['QuantizedConv2d']
MaxPool2d=['MaxPool2d']
shortcut=['QFunctional']
shortcut_from=-3
ReLU=['ReLU']
softmax=['softmax','logsoftmax']
shortcut_layer_name=['shortcut','downsample']
def make_net_dict(input_width,input_height,input_channel,mean_list,std_list):
    block_dict= dict()
    block_dict["layer_type"]="[net]"
    block_dict["batch"]="1"
    block_dict["subdivisions"]="1"
    block_dict["max_batches"]="1"
    block_dict["momentum"]="0.9"
    block_dict["decay"]="0.00005"
    block_dict["policy"]="poly"
    block_dict["power"]="4"
    block_dict["learning_rate"]="0.01"
    block_dict["angle="]="1"
    block_dict["hue="]="1"
    block_dict["saturation"]="1"
    block_dict["exposure"]="1"
    block_dict["aspect"]="1"
    block_dict["height"]=str(input_height)
    block_dict["width"]=str(input_width)
    block_dict["channels"]=str(input_channel)
    block_dict["quantization_type"]=str(0)
    block_dict["start_check_point"]=str(0)
    block_dict["end_check_point"]=str(0)
    block_dict["input_scale"]=str(0)
    block_dict["input_zeropoint"]=str(0)
    block_dict["normalize_mean_0"]=str(mean_list[0])
    block_dict["normalize_mean_1"]=str(mean_list[1])
    block_dict["normalize_mean_2"]=str(mean_list[2])
    block_dict["normalize_var_0"]=str(std_list[0])
    block_dict["normalize_var_1"]=str(std_list[1])
    block_dict["normalize_var_2"]=str(std_list[2])
    return block_dict


def make_layer_dict(module,model_dict):
    block_dic=dict()
    layer_num=model_dict["layer_count"]+1
    
    if module._get_name() in Quantize:
        model_dict['net']['input_scale']=str(module.scale.numpy())[1:-1]
        model_dict['net']['input_zeropoint']=str(module.zero_point.numpy())[1:-1]
        model_dict['net']['quantization_type']="1"
        
    if module._get_name() in QuantizedConvReLU2d:
        model_dict["layer_count"]+=1
        layer_num=model_dict["layer_count"]
        block_dict= dict()
        block_dict["layer_type"]="[convolutional]"
        block_dict["quantization_type"] ="1"
        block_dict["quantization_layer_scale"] =str(module.scale)
        block_dict["quantization_layer_zeropoint"] =str(module.zero_point)
        block_dict["filters"] =str(module.out_channels)
        block_dict["size"] =str(module.kernel_size[0])
        block_dict["pad"] =str(module.padding[0])
        block_dict["stride"] =str(module.stride[0])
        block_dict["activation"] ="relu"
        
        model_dict["before_layer_type"]=module._get_name()
        model_dict["before_layer_quantization_type"]=block_dict["quantization_type"]
        model_dict["before_layer_zeropoint"]=block_dict["quantization_layer_zeropoint"]
        model_dict["before_layer_scale"]=block_dict["quantization_layer_scale"]       
        model_dict[layer_num]=block_dict
    if module._get_name() in QuantizedConv2d:
        model_dict["layer_count"]+=1
        layer_num=model_dict["layer_count"]
        block_dict= dict()
        block_dict["layer_type"]="[convolutional]"
        block_dict["quantization_type"] ="1"
        block_dict["quantization_layer_scale"] =str(module.scale)
        block_dict["quantization_layer_zeropoint"] =str(module.zero_point)
        block_dict["filters"] =str(module.out_channels)
        block_dict["size"] =str(module.kernel_size[0])
        block_dict["pad"] =str(module.padding[0])
        block_dict["stride"] =str(module.stride[0])
        block_dict["activation"] ="linear"
        
        model_dict["before_layer_type"]=module._get_name()
        model_dict["before_layer_quantization_type"]=block_dict["quantization_type"]
        model_dict["before_layer_zeropoint"]=block_dict["quantization_layer_zeropoint"]
        model_dict["before_layer_scale"]=block_dict["quantization_layer_scale"]       
        model_dict[layer_num]=block_dict

    if module._get_name() in ReLU:
        layer_num=model_dict["layer_count"]
        if model_dict["before_layer_type"] in QuantizedConv2d:
            model_dict[layer_num]["activation"] = "relu"
        if model_dict["before_layer_type"] in QuantizedLinear:
            model_dict[layer_num]["activation"] = "relu" 

    if module._get_name() in MaxPool2d: 
        model_dict["layer_count"]+=1
        layer_num=model_dict["layer_count"]
        block_dict= dict()    
        block_dict["layer_type"]="[maxpool]"
        block_dict["quantization_type"] =model_dict["before_layer_quantization_type"]
        block_dict["quantization_layer_scale"] =model_dict["before_layer_scale"]
        block_dict["quantization_layer_zeropoint"] =model_dict["before_layer_zeropoint"]
        block_dict["size"] =str(module.kernel_size)
        block_dict["pad"] =str(module.padding)
        block_dict["stride"] =str(module.stride)
        model_dict["before_layer_type"]=module._get_name()
        model_dict[layer_num]=block_dict
        
    if module._get_name() in shortcut:
        model_dict["layer_count"]+=1  
        layer_num=model_dict["layer_count"]
        block_dict= dict()
        block_dict["layer_type"]="[shortcut]"
        block_dict["quantization_type"] =model_dict["before_layer_quantization_type"]
        block_dict["quantization_layer_scale"] =model_dict["before_layer_scale"]
        block_dict["quantization_layer_zeropoint"] =model_dict["before_layer_zeropoint"]
        block_dict["from"] =str(shortcut_from)
        
        model_dict["before_layer_type"]=module._get_name()
        model_dict[layer_num]=block_dict
        
    if module._get_name() in QuantizedLinear:
        model_dict["layer_count"]+=1  
        layer_num=model_dict["layer_count"]
        block_dict= dict() 
        block_dict["layer_type"]="[connected]"
        block_dict["quantization_type"] ="1"
        block_dict["quantization_layer_scale"] =str(module.scale)
        block_dict["quantization_layer_zeropoint"] =str(module.zero_point)
        block_dict["output"] =str(module.out_features)
        model_dict["before_layer_type"]=module._get_name()
        block_dict["activation"] ="linear"
        model_dict[layer_num]=block_dict
        
    if module._get_name() in softmax:
        model_dict["layer_count"]+=1  
        layer_num=model_dict["layer_count"]
        block_dict= dict() 
        block_dict["layer_type"]="[softmax]"
        block_dict["groups"]="1"
        block_dict["quantization_type"] =model_dict["before_layer_quantization_type"]
        model_dict["before_layer_type"]=module._get_name()   
        model_dict[layer_num]=block_dict
    return model_dict #,last_top_layer

def iter_module_children_recur_cfg(model,model_dict):
    for module_name, module in model.named_children():
        model_dict=make_layer_dict(module,model_dict)
        if module_name not in shortcut_layer_name:
            iter_module_children_recur_cfg(module,model_dict)
def make_cfg_from_pytorch(model,cfgfile_name,input_width,input_height,input_channel,mean_list,std_list):
    
    model_dict=dict()
    model_dict["net"]=make_net_dict(input_width,input_height,input_channel,mean_list,std_list)
    model_dict["layer_count"]=-1
    model_dict["before_layer_type"]=0
    model_dict["before_layer_quantization_type"]=0
    model_dict["before_layer_zeropoint"]=0
    model_dict["before_layer_scale"]=0
    
    last_top_layer=-1
    iter_module_children_recur_cfg(model,model_dict)
    # for module_name, module in model.named_children():
    #     model_dict=make_layer_dict(module,model_dict)
    #     if module_name not in shortcut_layer_name:
    #         for module_name_, module_ in module.named_children():
    #             model_dict=make_layer_dict(module_,model_dict)
    #             if module_name_ not in shortcut_layer_name:        
    #                 for module_name__, module__ in module_.named_children():       
    #                     make_layer_dict(module__,model_dict)
    #                     if module_name__ not in shortcut_layer_name:                    
    #                         for module_name___, module___ in module__.named_children():            
    #                             make_layer_dict(module___,model_dict)
    #                             if module_name___ not in shortcut_layer_name:
    #                                 for module_name____, module____ in module___.named_children():   
    #                                     make_layer_dict(module____,model_dict)
    if(model_dict["before_layer_type"] not in softmax):                      
        model_dict["layer_count"]+=1  
        layer_num=model_dict["layer_count"]
        block_dict= dict() 
        block_dict["layer_type"]="[softmax]"
        block_dict["groups"]="1"
        block_dict["quantization_type"] =model_dict["before_layer_quantization_type"]
        model_dict["before_layer_type"]=softmax      
        model_dict[layer_num]=block_dict
    write_line=''
    with open(cfgfile_name, 'wb') as cfgfile:
        for layer in model_dict:
            if type(model_dict[layer]) is dict:
                for layer_options in model_dict[layer]:
                    if(layer_options=="layer_type"):                    
                        if model_dict[layer][layer_options]!="[net]":
                            write_line="\n"
                        write_line= write_line+model_dict[layer][layer_options]+"\n"
                        print(write_line)
                        cfgfile.write(write_line.encode('utf-8'))
                    elif(layer_options!="layer_num"):
                        write_line=layer_options+"="+model_dict[layer][layer_options]+"\n"
                        print(write_line)
                        cfgfile.write(write_line.encode('utf-8'))
        
    
    
    