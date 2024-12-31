class Config {
    static version;
    static baseUrl;
    
    // Static block
    static {
      this.version = "1.0.0";
      this.baseUrl = process.env.BASE_URL || "https://default.url";
      console.log(`Initialized Config: version=${this.version}, baseUrl=${this.baseUrl}`);
    }

    static incrementCounter() {
        this.counter++;
    }    
  }

  // Usage
  console.log(Config.version); // "1.0.0"
  console.log(Config.baseUrl); // "https://default.url" (or the value of process.env.BASE_URL)
  