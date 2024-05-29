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

    @Test
    public void testParser() {
        try {
            String content = loadContent("helloworld.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertTrue(visitor.getTotalStatements().get() > 0);
            assertNotNull(p);
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }
    
    @Test
    public void testStatements() {
        try {
            String content = loadContent("examples/mysql.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(174, visitor.getTotalStatements().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }
    
    @Test
    public void testArrowFunctions() {
        try {
            String content = loadContent("examples/ArrowFunctions.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(5, visitor.getTotalArrowDeclarations().get());
        } catch (Exception e) {
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
            assertEquals(12, visitor.getTotalAsyncDeclarations().get());
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
            assertEquals(4, visitor.getTotalAwaitDeclarations().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }
    
    @Test
    public void testNullCoalesce() {
        try {
            String content = loadContent("examples/NullCoalesce.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(5, visitor.getTotalNullCoalesceOperators().get());
        } catch (Exception e) {
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
            assertEquals(7, visitor.getTotalArrayDestructuring().get());
            assertEquals(7, visitor.getTotalObjectDestructuring().get());
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
            assertEquals(7, visitor.getTotalDefaultParameters().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testSpreadArguments() {
        try {
            String content = loadContent("examples/SpreadAndRest.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(6, visitor.getTotalSpreadArguments().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testRest() {
        try {
            String content = loadContent("examples/SpreadAndRest.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(2, visitor.getTotalRestStatements().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testPromise() {
        try {
            String content = loadContent("examples/Promises.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(2, visitor.getTotalNewPromises().get());
            assertEquals(3, visitor.getTotalPromiseAllAndThenIdiom().get());
        } catch (Exception e) {
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
            assertEquals(6, visitor.getTotalLetDeclarations().get());
        } catch (Exception e) {
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
            assertEquals(5, visitor.getTotalConstDeclaration().get());
        } catch (Exception e) {
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
            assertEquals(7, visitor.getTotalExportDeclarations().get());
            assertEquals(13, visitor.getTotalImportStatements().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }
    
    @Test
    public void testNumericLieralSeparator() {
    	try {
    		String content = loadContent("examples/NumericSeparator.js");
    		JavaScriptParser.ProgramContext p = parser.parse(content);
    		JSVisitor visitor = new JSVisitor();
    		p.accept(visitor);
    		assertEquals(10, visitor.getTotalNumericLiteralSeparators().get());
    	} catch (Exception e) {
    		e.printStackTrace();
    		fail();
    	}
    }
    @Test
    public void testBigInt() {
    	try {
    		String content = loadContent("examples/BigIntLiteral.js");
    		JavaScriptParser.ProgramContext p = parser.parse(content);
    		JSVisitor visitor = new JSVisitor();
    		p.accept(visitor);
            content = loadContent("examples/mysql.js");
    		p = parser.parse(content);
    		p.accept(visitor);
    		assertEquals(8, visitor.getTotalBigInt().get());
    	} catch (Exception e) {
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
            assertEquals(15, visitor.getTotalClassDeclarations().get());
        } catch (Exception e) {
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
            assertEquals(11, visitor.getTotalYieldDeclarations().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }
    @Test
    public void testHashBangComment() {
    	try {
    		String content = loadContent("examples/HashBangComment.js");
    		JavaScriptParser.ProgramContext p = parser.parse(content);
    		JSVisitor visitor = new JSVisitor();
    		p.accept(visitor);
    		assertEquals(1, visitor.getTotalHashBangLines().get());
    	} catch (Exception e) {
    		e.printStackTrace();
    		fail();
    	}
    }

    @Test
    public void testOptionalChain() {
        try {
            String content = loadContent("examples/OptionalChain.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(6, visitor.getTotalOptionalChain().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testTemplateLiterals() {
        try {
            String content = loadContent("examples/TemplateLiterals.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(4, visitor.getTotalTemplateStringExpressions().get());

            content = loadContent("examples/TemplateStrings.js");
            p = parser.parse(content);
            visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(8, visitor.getTotalTemplateStringExpressions().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testObjectProperties() {
        try {
            String content = loadContent("examples/EnhancedObjectProperties.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(4, visitor.getTotalObjectProperties().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testComputedProperties() {
        try {
            String content = loadContent("examples/EnhancedObjectProperties.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(5, visitor.getTotalComputedProperties().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testRegularExpressions() {
        try {
            String content = loadContent("examples/EnhancedRegularExpression.js");
            JavaScriptParser.ProgramContext p = parser.parse(content);
            JSVisitor visitor = new JSVisitor();
            p.accept(visitor);
            assertEquals(2, visitor.getTotalRegularExpressions().get());
        } catch (Exception e) {
            e.printStackTrace();
            fail();
        }
    }
    @Test
    public void testExponentiationAssigments() {
    	try {
    		String content = loadContent("examples/ExponentiationAssignment.js");
    		JavaScriptParser.ProgramContext p = parser.parse(content);
    		JSVisitor visitor = new JSVisitor();
    		p.accept(visitor);
    		assertEquals(6, visitor.getTotalExponentiationAssignments().get());
    	} catch (Exception e) {
    		e.printStackTrace();
    		fail();
    	}
    }

    @Ignore
    public void testParserReact() throws Exception {
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
        int totalArrowDeclarations = 0;
        int totalAsyncDeclarations = 0;
        int totalStatements = 0;

        for (File file : files) {
            try {
                String content = new String(Files.readAllBytes(file.toPath()));
                JavaScriptParser.ProgramContext p = parser.parse(content);
                assertNotNull(p);
                JSVisitor visitor = new JSVisitor();
                p.accept(visitor);
                totalArrowDeclarations += visitor.getTotalArrowDeclarations().incrementAndGet();
                totalAsyncDeclarations += visitor.getTotalAsyncDeclarations().incrementAndGet();
                totalStatements += visitor.getTotalStatements().incrementAndGet();
            } catch (Throwable e) {
                errors++;
                System.err.println(e.getMessage());
            }
        }

        assertEquals(47, files.size());
        assertEquals(97, totalArrowDeclarations);
        assertEquals(62, totalAsyncDeclarations);
        assertEquals(1341, totalStatements);
        assertEquals(1, errors);
    }
}
