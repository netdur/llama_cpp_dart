class Note {
  String title;
  String content;

  Note({required this.title, required this.content});

  @override
  String toString() {
    return 'Note(title: "$title", content: "$content")';
  }
}

List<Note> notes = [
  // Science Notes
  Note(
      title: 'Photosynthesis Basics',
      content:
          'Process plants use to convert light energy into chemical energy (glucose). Inputs: CO2, Water, Light. Outputs: Glucose, Oxygen. Occurs in chloroplasts.'),
  Note(
      title: 'Newton\'s First Law',
      content:
          'An object at rest stays at rest, and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force. Also known as the law of inertia.'),
  Note(
      title: 'pH Scale Reminder',
      content:
          'Measures acidity/alkalinity. <7 = Acidic, 7 = Neutral, >7 = Basic (Alkaline). It\'s a logarithmic scale.'),
  Note(
      title: 'Mitochondria Function',
      content:
          '"Powerhouse of the cell." Generates most of the cell\'s supply of ATP (adenosine triphosphate), used as chemical energy. Site of cellular respiration.'),

  // Math Notes
  Note(
      title: 'Quadratic Formula',
      content:
          'For ax² + bx + c = 0, the solution is x = [-b ± sqrt(b² - 4ac)] / 2a. Remember the discriminant (b² - 4ac) determines the nature of the roots.'),
  Note(
      title: 'Area of a Circle',
      content:
          'A = πr², where r is the radius of the circle. π (pi) is approximately 3.14159.'),

  // History Notes
  Note(
      title: 'D-Day Landing Date',
      content:
          'June 6, 1944. Allied invasion of Normandy during World War II. Operation Overlord. Major turning point on the Western Front.'),
  Note(
      title: 'Fall of Western Roman Empire',
      content:
          'Traditionally dated to 476 AD when the last Western Roman Emperor, Romulus Augustulus, was deposed by the Germanic chieftain Odoacer.'),

  // Literature / Language Notes
  Note(
      title: '"To be or not to be" Source',
      content:
          'From Shakespeare\'s Hamlet, Act 3, Scene 1. Spoken by Hamlet. Famous soliloquy contemplating life, death, and suicide.'),
  Note(
      title: 'There vs. Their vs. They\'re',
      content:
          'There: Place (Go over there). Their: Possessive pronoun (Their books). They\'re: Contraction of "they are" (They\'re happy).'),
  Note(
      title: 'Simile vs. Metaphor',
      content:
          'Simile: Comparison using "like" or "as" (Brave as a lion). Metaphor: Direct comparison stating one thing *is* another (The world is a stage).'),

  // Computer Science / Tech Notes
  Note(
      title: 'Dart `late` Keyword',
      content:
          'Used for non-nullable variables initialized *after* their declaration. Promises the compiler it will be initialized before use. Can also be used for lazy initialization.'),
  Note(
      title: 'List vs. Set (Data Structures)',
      content:
          'List: Ordered collection, allows duplicates. Access by index O(1) usually. Set: Unordered (usually), *no* duplicates allowed. Fast `contains` checks O(1) on average.'),
  Note(
      title: 'HTTP GET vs POST',
      content:
          'GET: Requests data from a resource. Parameters sent in URL. Idempotent. POST: Submits data to be processed (e.g., form submission). Data sent in request body. Not idempotent.'),
  Note(
      title: '`git commit -m` Usage',
      content:
          'Command to save staged changes to the local repository. The `-m` flag allows adding a concise commit message directly on the command line.'),

  // General / Other Notes
  Note(
      title: 'Capital of Australia',
      content: 'Canberra. (Common misconceptions: Sydney or Melbourne).'),
  Note(
      title: 'Impressionism Key Artists',
      content:
          'Monet, Renoir, Degas, Pissarro, Morisot. Focused on capturing light and fleeting moments, often painted outdoors (\'en plein air\').'),
  Note(
      title: 'Pomodoro Technique',
      content:
          'Time management method. Work in focused 25-minute intervals separated by short (5-min) breaks. Longer breaks after 4 \'pomodoros\'. Helps maintain concentration.'),
  Note(
      title: 'Roux Basics (Cooking)',
      content:
          'Thickening agent made from equal parts fat (usually butter) and flour, cooked together. Base for many classic sauces (béchamel, velouté, espagnole).'),
  Note(
      title: 'Meeting Follow-up: Project Alpha',
      content:
          'Action Items: 1. Send draft proposal to Jane by EOD Friday. 2. Schedule follow-up sync for next Tuesday. 3. Research vendor options for component X.'),
];
