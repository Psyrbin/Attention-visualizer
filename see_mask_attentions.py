import sys
import subprocess
import numpy as np
from cgi import escape

input_file = sys.argv[1]
bert = sys.argv[2]
if len(sys.argv) > 3:
    layers = sys.argv[3]
else:
    layers = '0,1,2,3,4,5,6,7,8,9,10,11'

layer_idxs = [int(x) for x in layers.split(',')]

with open(input_file, 'r') as f:
    for line in f:
        text = line
        break

args = ['python', 'bert/extract_features.py']
args.append('--input_file=' + input_file)
args.append('--output_file=')
args.append('--vocab_file=' + bert + 'vocab.txt')
args.append('--bert_config_file=' + bert + 'bert_config.json')
args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
args.append('--layers=' + layers)
args.append('--max_seq_length=128')
args.append('--batch_size=8')
args.append('--do_lower_case=False')
args.append('--attention=True')
args.append('--mask_underscore=True')
subprocess.run(args)

tokens = []
with open('tokens.txt', 'r') as f:
    for i, line in enumerate(f):
        tokens.append(line[:-1])
        if line[:-1] == '[MASK]':
            mask_idx = i
print(mask_idx)

with open('result.html', 'w') as res:
    res.write('<HTML>\n')
    for idx in layer_idxs:
        res.write('<div>Layer ' + str(idx) + '</div><br>\n')
        layer = np.load('_layer_' + str(idx) + '.npy')
        for head in range(layer.shape[0]):
            res.write('<div>HEAD ' + str(head) + '</div>\n')
            attentions = layer[head, mask_idx,:]
            for word_idx, attn in enumerate(attentions):
                if attn < 0:
                    color = 'rgba(250, 10, 10, ' + str(-attn)
                else:
                    color = 'rgba(10, 10, 250, ' + str(attn)
                res.write('<span style="background-color:' + color + ');">' + escape(tokens[word_idx]) + '</span>')
            res.write('<br><br>\n')
        res.write('<br>\n')

    res.write('</HTML>')