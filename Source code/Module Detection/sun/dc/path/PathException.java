package sun.dc.path;
 
 public class PathException extends java.lang.Exception
 {
 
     public static final String
         BAD_PATH_endPath = "endPath: bad path",
         BAD_PATH_useProxy = "useProxy: bad path",
         DUMMY = "";
 
     // Constructors
     public PathException() {
         super();
     }
 
     public PathException(String s) {
         super(s);
     }
 }
