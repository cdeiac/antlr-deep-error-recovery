package com.antlr.data;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.InputStream;

public class Dataset {
    
    private String fileName;
    private BufferedInputStream inputStream;
    private BufferedOutputStream outputStream;


    public void init(String fileName) {
        // load file and save stream
        this.inputStream  = this.loadFile(fileName);
  
    }

    /**
     *  Loads the dataset from classpath
     * 
     *  @param fileName the name of the file (incl. file extension)
     */
    private BufferedInputStream loadFile(String fileName) {
        InputStream is = this.getClass().getClassLoader().getResourceAsStream(fileName);
        return new BufferedInputStream(is); 
    }

    /**
     *  Save modified dataset
     */
    private void saveFile() {

        if (this.inputStream == null) {
            return;
        }
        
    }

    
    public BufferedInputStream getInputStream() {
        return this.inputStream;
    }

    
    public BufferedOutputStream getOutputStream() {
        return this.outputStream;
    }


}
