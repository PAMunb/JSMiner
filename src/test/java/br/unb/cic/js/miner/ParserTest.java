package br.unb.cic.js.miner;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;


public class ParserTest {

    private JSParser parser;

    @Before
    public void setUp() {
        parser = new JSParser();
    }


    private String loadContent(String file) throws IOException, URISyntaxException {
        ClassLoader classLoader = getClass().getClassLoader();
        String content = new String(Files.readAllBytes(Paths.get(classLoader.getResource(file).toURI())));
        return content;
    }

	@Ignore
    public void testParser() {
        try {
            String content = loadContent("helloworld.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertTrue(visitor.getTotalArrowDeclarations() > 0);
            assertNotNull(p);
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testStatments() {
        try {
            String content = loadContent("examples/ArrowFunctions.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(10, visitor.getTotalStatements());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }
    
	@Test
    public void testAsync() {
        try {
            String content = loadContent("examples/AsyncAwait.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(11, visitor.getTotalAsyncDeclarations());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }
	
	@Test
	public void testDestructuring() {
		try {
			String content = loadContent("examples/DestructuringAssignment.js");
	        JavaScriptParser.ProgramContext p = parser.parse(content);
	        JSVisitor visitor = new JSVisitor();
	        p.accept(visitor);
	        assertEquals(3, visitor.getTotalArrayDestructuring());
	        assertEquals(3, visitor.getTotalObjectDestructuring());
		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}

	}
	
	@Test
	public void testDefaultParameters() {
		try {
			String content = loadContent("examples/DefaultParameter.js");
	        JavaScriptParser.ProgramContext p = parser.parse(content);
	        JSVisitor visitor = new JSVisitor();
	        p.accept(visitor);
	        assertEquals(1, visitor.getTotalDefaultParameters());
		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}
	}
	
	@Test
	public void testSpreadArguments() {
		try {
			String content = loadContent("examples/Spread.js");
	        JavaScriptParser.ProgramContext p = parser.parse(content);
	        JSVisitor visitor = new JSVisitor();
	        p.accept(visitor);
	        assertEquals(1, visitor.getTotalSpreadArguments());
		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}	
	}

    @Test
    public void testPromise(){
        try {
            String content = loadContent("examples/Promises.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(2, visitor.getTotalNewPromises());
            assertEquals(1, visitor.getTotalPromiseAllAndThenIdiom());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testAwait() {
        try {
            String content = loadContent("examples/AsyncAwait.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(3, visitor.getTotalAwaitDeclarations());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testLet() {
        try {
            String content = loadContent("examples/Scoping.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(6, visitor.getTotalLetDeclarations());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testConst() {
        try {
            String content = loadContent("examples/Constants.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(5, visitor.getTotalConstDeclaration());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testModules() {
        try {
            String content = loadContent("examples/Modules.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(5, visitor.getTotalExportDeclarations());
            assertEquals(8, visitor.getTotalImportStatements());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testRest() {
        try {
            String content = loadContent("examples/Rest.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(1, visitor.getTotalRestStatements());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testClass() {
        try {
            String content = loadContent("examples/Classes.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(15, visitor.getTotalClassDeclarations());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testYield() {
        try {
            String content = loadContent("examples/Generators.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(6, visitor.getTotalYieldDeclarations());
        }
        catch(Exception e) {
            e.printStackTrace();
            fail();
        }
    }

	@Ignore
    public void testParserReact() throws Exception{
    	ClassLoader classLoader = getClass().getClassLoader();
        URI directoryPath = classLoader.getResource("examples").toURI();
        
        List<File> files = new ArrayList<>();

		Files.walk(Paths.get(directoryPath)).forEach(filePath -> {
            if (Files.isRegularFile(filePath) && filePath.toString().endsWith(".js")) {
            	files.add(filePath.toFile());
            }
        });
		
		assertFalse(files.isEmpty());

        int errors = 0;
		int totalFunctionDeclarations = 0;
		int totalAsyncDeclarations = 0;

        for (File file : files) {
			try {
	            String content = new String(Files.readAllBytes(file.toPath()));
	            JavaScriptParser.ProgramContext p = parser.parse(content);
	            assertNotNull(p);
	            JSVisitor visitor = new JSVisitor();
	            p.accept(visitor);
	            totalFunctionDeclarations += visitor.getTotalArrowDeclarations();
	            totalAsyncDeclarations += visitor.getTotalAsyncDeclarations();
			} catch (Throwable e) {
				errors++;
			}
		}

        assertEquals(34, files.size());
        assertEquals(63, totalFunctionDeclarations);
        assertEquals(65, totalAsyncDeclarations);
        assertEquals(0, errors);
    }
}
