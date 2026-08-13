using System;
using System.Collections.Generic;

public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }

    public Person(string name, int age)
    {
        Name = name;
        Age = age;
    }

    public string Describe()
    {
        return $"{Name} is {Age} years old.";
    }
}

public class Team
{
    private readonly List<Person> _members = new();

    public void AddMember(Person member)
    {
        _members.Add(member);
    }

    public void PrintMembers()
    {
        foreach (var member in _members)
        {
            Console.WriteLine(member.Describe());
        }
    }
}

public class GreetingService
{
    public string BuildWelcomeMessage(string name)
    {
        return $"Welcome, {name}! We are glad you are here.";
    }

    public void Greet(Person person)
    {
        Console.WriteLine(BuildWelcomeMessage(person.Name));
    }
}

public class Program
{
    public static void Main()
    {
        Console.WriteLine(">>>> Hello, World! >>>>>");

        var team = new Team();
        team.AddMember(new Person("Ada", 29));
        team.AddMember(new Person("Grace", 34));
        team.AddMember(new Person("Linus", 54));

        Console.WriteLine("\nTeam members:");
        team.PrintMembers();

        var greeter = new GreetingService();
        var person = new Person("Berlin", 31);

        Console.WriteLine();
        greeter.Greet(person);

        Console.WriteLine("\nA little more code, a little better demo.");
    }
}
