let jsonData;
try {
    jsonData = JSON.parse(str); // (A)
} catch {
    jsonData = DEFAULT_DATA;
}


function logId(person) {
    let id = 'No ID';
    try {
        id = person.data.id;
    } catch {}
    console.log(id);
}

//não deve ser contado
try {
    jsonData = JSON.parse(str);
} catch (err) {
    if (err instanceof SyntaxError) {
        jsonData = DEFAULT_DATA;
    } else {
        throw err;
    }    
}