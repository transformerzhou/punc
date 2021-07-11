#!/usr/bin/env python
# coding: utf-8

# In[31]:


# mode = 'test'

for mode in ['train', 'dev', 'test']:

    lines = open('bcut/bcut.{}.txt'.format(mode)).readlines()


    # In[32]:


    def text_process(texts):
        lines = []
        texts = texts.strip().split('[END]')
        line = ''
        #断句，最大长度为200
        for text in texts:
            if not text.strip():
                continue
            if len(line+text)<200:
                line += text+'，'
            else:
                lines.append(line)
                line = text+'，'
    #             print(lines[-1])
        if line != '，':
            lines.append(line)
    #     print(lines, len(lines))
        if len(lines) == 1 and not lines[0].endswith(' ，'):
            return [lines[0][:-1] + ' ，']
        return lines

    # 写入新文件
    with open('{}.txt'.format(mode), 'w') as f:
        for line in lines:
            for cut in text_process(line):
                if 10<len(cut)<500:
                    f.write(cut+'\n')








