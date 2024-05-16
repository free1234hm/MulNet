 package sun.dc.path;
 
 public class PathError extends java.lang.RuntimeException
 {
 
     public static final String
         UNEX_beginPath = "beginPath: unexpected",
         UNEX_beginSubpath = "beginSubpath: unexpected",
         UNEX_appendLine = "appendLine: unexpected",
         UNEX_appendQuadratic = "appendQuadratic: unexpected",
         UNEX_appendCubic = "appendCubic: unexpected",
         UNEX_closedSubpath = "closedSubpath: unexpected",
         UNEX_endPath = "endPath: unexpected",
         UNEX_useProxy = "useProxy: unexpected",
         UNEX_getBox = "getBox: unexpected",
         UNEX_sendTo = "sendTo: unexpected",
         BAD_boxdest = "getBox: invalid box destination array",
         BAD_pathconsumer = "sendTo: invalid path consumer",
         INTERRUPTED = "",
         DUMMY = "";
 
     // Constructors
     public PathError() {
        super();
    }

    public PathError(String s) {
        super(s);
     }
 }
