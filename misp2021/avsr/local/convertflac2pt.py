import os
import soundfile as sf
import torch
import multiprocessing

import codecs
def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content)]
        with codecs.open(textpath, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None

def process_files(file_list, output_path):
    for filename in file_list:
        uid,path = filename.split(" ")
        audio, sr = sf.read(path)
        tensor = torch.from_numpy(audio)*2**15 
        tensor = tensor.type(torch.int16) # pay attention
        output_file = os.path.join(output_path, uid + '.pt')
        torch.save(tensor, output_file)


if __name__ == '__main__':
    import sys 
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_files = text2lines(input_file)

    # Set the number of processes to use
    num_processes = 32
    # Split the input file list into chunks for each process to handle
    lines_chunks = [input_files[i::num_processes] for i in range(num_processes)]
    # Create and start the processes
    processes = []
    for i in range(num_processes):
        suboutput_dir = os.path.join(output_dir,str(i))
        if not os.path.exists(suboutput_dir):
            os.mkdir(suboutput_dir)
        process = multiprocessing.Process(target=process_files, args=(lines_chunks[i], suboutput_dir))
        print(suboutput_dir)
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
